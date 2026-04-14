import warnings

warnings.filterwarnings(
    "ignore", message=".*`torch.cuda.amp.autocast.*` is deprecated.*"
)
import torch
from torch.utils.data import DataLoader
import sys

sys.path.append("../src")
from models import AutoEncoder
from utils import AeroDataset, inference_with_n_quantizer
import json

valid_path = "../mid_results/valid-set/*"

"""
This file is used for evaluations of models.
"""

model_path = ""  # path to a .pt file
path_to_config = ""  #  path to a .json file

with open(path_to_config, "r") as f:
    config = json.load(f)

code_book = config["code_book"]
quantizers = config["quantizers"]
batch_size = 128
window_size = 800
n_channels = 36
sample_ = 0
sensor_ = 15
workers_num_ = 8
dec_code = config["dec_code"]
enc_code = (
    1  # 2  # I should always be using 1 or 2 since GAP9 doesnt like quantized batchnorm
)
CUSTOM = False

print(config)


def aero_DataLoader(
    device: str, dataset: torch.utils.data.Dataset, workers_num: int, batch_size: int
):
    if device == "cpu":
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            pin_memory=False,
            num_workers=workers_num,
        )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=workers_num,
    )


def compute_center_error(model, dataloader, device="cpu"):
    """
    This function evaluates the models output in relation to the input on the inmost 512 values, and returns the average reconstruction error
    over the whole validation set.

    Args:
        model (nn.Module): The Autoencoder
        dataloader (DataLoader): The validation set prepared by aero_DataLoader()
        device (string, optional): Either 'cuda' or 'cpu', device to be used. Defaults to 'cpu'.

    Returns:
        float: Average error in % over the whole validation set
    """
    total_error = 0.0
    total_sq_error = 0.0
    total_input_sq = 0.0
    total_elements = 0
    with torch.no_grad():
        for x_batch, _ in dataloader:
            x_batch = x_batch.to(device)
            if CUSTOM:
                # outputs = inference_with_n_quantizer(model, x_batch, 4)
                outputs, _, _ = model(x_batch)
            else:
                outputs, _, _ = model(x_batch)

            center_start = (x_batch.size(-1) - 512) // 2
            center_end = center_start + 512
            input_center = x_batch[:, :, center_start:center_end]
            output_center = outputs[:, :, center_start:center_end]

            # Error
            epsilon = 1e-8  # Small value to handle division by zero
            abs_diff = (output_center - input_center).abs()

            percentage_error = (abs_diff / (input_center.abs() + epsilon)) * 100

            total_error += percentage_error.sum().item()
            total_sq_error += ((output_center - input_center) ** 2).sum().item()
            total_input_sq += (input_center**2).sum().item()
            total_elements += (
                percentage_error.numel()
            )  # this is for batches, otherwise could just use 512

    avg_error = total_error / total_elements
    nmse_error = (total_sq_error / (total_input_sq + epsilon)) * 100
    mse_error = total_sq_error / total_elements
    return avg_error, nmse_error, mse_error


if __name__ == "__main__":
    device_ = "cpu"  # torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    model = AutoEncoder(
        arch_id="ss00",
        c_in=n_channels,
        RVQ=True,
        codebook_size=code_book,
        quantizers=quantizers,
        dec_code=dec_code,
        enc_code=enc_code,
    )

    checkpoint = torch.load(model_path, map_location=device_)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
        print(f"Checkpoint loaded from {model_path}")
    else:
        model.load_state_dict(checkpoint)
        print(f"State dict loaded directly from {model_path}")

    model.eval()

    aero_dataset_valid = AeroDataset(valid_path, device=device_)

    aero_dl_valid = aero_DataLoader(
        device=device_,
        dataset=aero_dataset_valid,
        workers_num=workers_num_,
        batch_size=128,
    )

    avgerr, nmse_error, mse_error = compute_center_error(model, aero_dl_valid, device_)
    print("Average error:", avgerr)
    print("Normalized MSE (NMSE) of model:", nmse_error)
    print("MSE per element:", mse_error)

