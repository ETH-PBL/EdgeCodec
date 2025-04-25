import numpy as np 
import pandas as pd
# To save the trained model
from csv import DictWriter
import random
import os
from simple_term_menu import TerminalMenu
from dataclasses import dataclass, field
import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset


import torch 
import torch.nn as nn

import shutup 
shutup.please()


MSE = lambda x, x_hat: np.mean(np.square(x - x_hat), axis = 1)


def generate_hexadecimal() -> str:
    hex_num  =  hex(random.randint(0, 16**16-1))[2:].upper().zfill(16)
    hex_num  =  hex_num[:4] + ":" + hex_num[4:8] + ":" + hex_num[8:12] + ":" + hex_num[12:]
    return hex_num

def is_file_exist(file_name: str) -> bool:
    return os.path.isfile(file_name)

def write_to_csv(data: dict) -> None: 
    write_dir = "../training_results.csv"
    if (is_file_exist(write_dir)): 
        append_to_csv(data)
    else: 
        create_csv(data)

def append_to_csv(data: dict) -> None:
    
    print("Appending to the training results csv file")
    print("++"*15)
    with open("../training_results.csv", "a") as FS:
        
        headers = list(data.keys())

        csv_dict_writer = DictWriter(FS, fieldnames = headers) 

        csv_dict_writer.writerow(data)

        FS.close()


def create_csv(data: dict) -> None:
    df = pd.DataFrame.from_dict(data, orient = "index").T.to_csv("../training_results.csv", header = True, index = False)
    print("Created the csv file is as follows:")
    print(df)


def updateMSE(model_id, mse):
    df = pd.read_csv("../training_results.csv")
    # check if the mse columns already exists
    if not 'mse' in df.columns:
        df['mse'] = pd.Series()
    
    df.loc[df["model_id"]==model_id, "mse"] = mse
    df.to_csv("../training_results.csv", index=False)

def modelChooser():
    df = pd.read_csv("../training_results.csv")
    options = []
    model_ids = []
    for row in df.iterrows():
        x = row[1]
        model_ids.append(x['model_id'])
        options.append(f"{x['model_id']} - {x['window_size']} -> {x['latent_channels']} x {x['latent_seq_len']}")
    terminal_menu = TerminalMenu(options)
    menu_entry_index = terminal_menu.show()

    model_id = model_ids[menu_entry_index]
    return df.loc[df['model_id'] == model_id].to_dict(orient='index')[menu_entry_index]

def checkNumber(string: str) -> int:
    integer = int(string)
    if integer in range(0, 10000):
        return integer
    raise Exception()


def read_matrix_from_file_raw(file_path):
    """
    Takes execution from the GVSOC and extracts the indices from the matrix. The gvsoc code to be used is corr_gvsoc_rvq.c, just save it in a .txt

    Args:
        file_path (string): Path to the save .txt from the GVSOC execution

    Returns:
        torch.tensor: Recovered indices
    """
    matrix = []
    with open(file_path, "r") as file:
        for line in file:
            line = line.strip()
            if not line or (not line.startswith("=") and not line.startswith("[")) or not line.endswith(", ];"):  # Skip invalid lines
                continue
            try:
                if line.startswith("="):
                    row = list(map(float, line[2:-4].split(", ")))
                else:
                    row = list(map(float, line[1:-4].split(", ")))
                matrix.append(row)
            except ValueError:
                print(f"Skipping invalid line: {line}")
                continue
    matrix = np.array(matrix)
    matrix = torch.tensor(matrix)
        
    return matrix


def rebuild_quantized(codebooks, indices, num):
    """
    This function rebuilds the latent space tensors from the given indices and codebooks, returns the tensor

    Args:
        codebooks (torch.tensor): The codebooks used by the model
        indices (torch.tensor): Indices provided by either the python inference or read_matrix_from_file_raw()
        num (int): With how many codebooks should the tensor be rebuilt

    Returns:
        torch.tensor: Reconstructed latent space tensor
    """
    batched = indices.dim() == 3
    embedding_dim = codebooks.size(2)
    if batched: # Batched
        batch_size, num_latents, num_quantizers = indices.shape
        result = torch.zeros(batch_size, num_latents, embedding_dim, device=codebooks.device)  
    elif indices.dim() == 2:
        indices = indices.long()
        num_latents, num_quantizers = indices.shape
        result = torch.zeros(num_latents, embedding_dim, device=codebooks.device)
        result = result.unsqueeze(0)
    else:
        print("Indices error", indices.dim())
        sys.exit()
    
    # Iterate over each quantizer
    for i in range(num): 
        if batched:
            result += codebooks[i, indices[:, :, i]]
        else:
            result += codebooks[i, indices[:, i]]
    return result

def inference_with_n_quantizer(model, x, quantizer_index=1):
    """
    Custom inference function to only use a select number of quantizers for reconstruction.

    Args:
        model (nn.Module): The autoencoder
        x (torch.tensor): Data to be processed
        quantizer_index (int, optional): How many quantizer to used. Defaults to 1.

    Returns:
        torch.tensor: Rebuilt tenspr
    """
    model.eval()
    
    with torch.no_grad():
        encoded = model.encoder(x)
        _, indices, _ = model.quantizer(encoded)
        reconstructed_vec = rebuild_quantized(model.quantizer.codebooks, indices, quantizer_index)
        decoded = model.decoder(reconstructed_vec)
    
    return decoded

class AeroDataset(Dataset):
    __doc__ = r"""
        This is an adaptation of the Dataset Class to only load batches in the 
        CPU/GPU RAM.  

        Given the method explained in <https://www.analyticsvidhya.com/blog/2021/09/torch-dataset-and-dataloader-early-loading-of-data/>
        for images, in this class, we implement a version for the time series data. 
        Notice that this is a naive adaptaion for time-series data types where 
        each sample should be stored in memory with a fixed sequence length.   
        In this case each input is [1x36x800]. 
        Since we are training an unsupervised Multi-variant AutoEncoder, the labels are the input data. 
        """ 
    def __init__(self, path: str, device: str, gpu_number:int = 0): 
        super().__init__()
        __doc__ = r"""
            Args: 
                path (str): Directory of the files
                device (str): Device that you want to load your data into. 
                gpu_number (int): if device is cuda, you should insert the number of GPU. 
                                    Notice that it is not active for the CPU case.
        """
        self.path = path 
        self.file_list = glob.glob(self.path)
        self.device = device 
        self.gpu_number = gpu_number
    def __getitem__(self, item):
        file_idx = self.file_list[item] 
        sample = torch.load(file_idx, map_location = lambda storage, loc:storage.cuda(self.gpu_number)) if self.device == 'cuda' else torch.load(file_idx, map_location = torch.device('cpu'))
        return sample , sample
    def __len__(self): 
        return len(self.file_list)
    
if __name__ == "__main__":
    
    
    answer = input("Do you want to save the Model? (y/n): ")
    if answer == "n": exit()

    answer = input("Please enter the latent channels? [0-100]: ")
    l_channels = checkNumber(answer)
    
    answer = input("Please enter the latent sequence length? [0-10000]: ")
    l_seq_len = checkNumber(answer)

    model_number = generate_hexadecimal()

    print("-"*50)
    print(f"Saving the model: {model_number}")

    

