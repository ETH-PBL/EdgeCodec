import torch.onnx
import torch
import torch.nn as nn 
from torchinfo import summary
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import sys
import os 
import time
import seaborn as sns

sys.path.append("../src")
from models import AutoEncoder
from utils import AeroDataset, inference_with_n_quantizer
import json

CLEAN = False
valid_path = "../mid_results/valid-set/*"

"""
This file is used for evaluations of models. This is rather extensive and modular but should work out of the box.
"""


model_path = '' # path to a .pt file
path_to_config = '' #  path to a .json file


with open(path_to_config, "r") as f:
    config = json.load(f)

code_book = config['code_book']
quantizers = config['quantizers']
batch_size = 128
window_size = 800
n_channels = 36
workers_num_ = 8
dec_code = config['dec_code']
enc_code = 2 
CUSTOM = False

def aero_DataLoader(device: str, dataset: torch.utils.data.Dataset, workers_num: int, batch_size: int):
    if device == "cpu":
        return  DataLoader(dataset, batch_size = batch_size, shuffle = True, pin_memory=False,num_workers = workers_num)
    return DataLoader(dataset, batch_size = batch_size, shuffle = True, pin_memory=True,num_workers = workers_num)

def compute_center_error(model, dataloader, device='cpu'):
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
    total_elements = 0
    with torch.no_grad():
        for x_batch, _ in (dataloader):
            x_batch = x_batch.to(device)
            if CUSTOM:
                outputs = inference_with_n_quantizer(model, x_batch, 4)
            else:
                outputs, _, _ = model(x_batch)

            center_start = (x_batch.size(-1) - 512) // 2
            center_end = center_start + 512
            input_center = x_batch[:, :, center_start:center_end]
            output_center = outputs[:, :, center_start:center_end]

            # Error
            epsilon = 1e-8  # Small value to handle division by zero
            abs_diff = (output_center - input_center).abs()

            # Calculate percentage error where input_center != 0
            percentage_error = torch.where(
                input_center.abs() > epsilon,  # Condition: input != 0
                (abs_diff / input_center.abs()) * 100,  # Percentage error
                abs_diff * 100  # Unsure about this case
            )


            total_error += percentage_error.sum().item()
            total_elements += percentage_error.numel() # this is for batches, otherwise could just use 512

    avg_error = total_error / total_elements
    return avg_error



if __name__ == "__main__": 

    device_ = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    model = AutoEncoder(arch_id = "ss00", c_in = n_channels, RVQ = True, 
                        codebook_size=code_book, quantizers=quantizers, dec_code = dec_code, enc_code = enc_code)


    model.eval()
    
    aero_dataset_valid = AeroDataset(valid_path, device = 'cpu')

    aero_dl_valid = aero_DataLoader(device= device_, 
                                    dataset= aero_dataset_valid, 
                                    workers_num=workers_num_, 
                                    batch_size=128)
    avgerr = compute_center_error(model, aero_dl_valid, device_)
    print("Average error of normal model:", avgerr)