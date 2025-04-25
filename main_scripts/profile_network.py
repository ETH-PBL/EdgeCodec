import torch.onnx
import torch
import torch.nn as nn 
import numpy as np
import sys
import os 
import time
import seaborn as sns
import pickle

sys.path.append("../src")
from models import AutoEncoder
import json
import onnxruntime as ort
import onnx
import numpy as np
from utils import read_matrix_from_file_raw, rebuild_quantized

"""
This file is used to do the cloud-side speed profiling.
"""


#device_ = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device_ = "cuda" if torch.cuda.is_available() else "cpu"
path_to_model = "" # path to a .pt file
path_to_config = "" # path to a .json file
path_for_warmup = '' # path to a .txt file for warmup 
path_to_indices = '' # path to a .txt file with the indices produced by the GVSOC C RVQ


def RVQ_pass(model, data):
    """
    Custom RVQ pass function.

    Args:
        model (nn.Module): The RVQ of the full model
        data (torch.tensor): data to be processed

    Returns:
        torch.tensor, torch.tensor: Rebuilt quantized output, indices for rebuilding
    """
    quantized, indices, _ , all_codes= model(data, return_all_codes = True)
    return quantized, indices
    

def prepare_model(code_book, quantizers, dec_code, enc_code):
    """
    Simple prepare model function, returns the loaded model.

    Args:
        code_book (int): How many code words per quantizer
        quantizers (_int): How many quantizers
        dec_code (int): Deciding what decoder
        enc_code (int): Deciding what encoder

    Returns:
        _type_: _description_
    """
    model = AutoEncoder(arch_id = "ss00", c_in = 36, RVQ = True, codebook_size = code_book, quantizers = quantizers, 
                        dec_code = dec_code, enc_code = enc_code)
    model.load_state_dict(torch.load(path_to_model))
    #model.to(device_)
    print("Loaded Model")
    return model

def warm_up(model, data):
    """
    Simple warm up function so that the  model parameters are allocated and ready to be used.

    Args:
        model (nn.Module): Model to be allocated
        data (torch.tensor): Dummy data
    """
    data = data.to(device_)
    for _ in range(100):
        output, _, _ = model.quantizer(data)
        _ = model.decoder(output)
    print("Warmup Complete")


def speed_check(model, codebooks):
    """
    Function to profile the rebuilding of the latent space data and the bass through the decoder.
    The rebuilding was done on CPU and the pass through of the decoder on gpu.

    Args:
        model (nn.Module): The whole autoencoder
        codebooks (Torch.Tensor): codebooks of the autoencoder

    Returns:
        float, float: cpu time, gpu time
    """
    start_time = time.time()
    actual_mat = read_matrix_from_file_raw(path_to_indices)
    actual_data = rebuild_quantized(codebooks, actual_mat, 4)
    actual_data = actual_data.to(device_)
    end_time = time.time()
    time_to_return = end_time - start_time
    
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    
    start_event.record()
    _ = model.decoder(actual_data)
    end_event.record()
    torch.cuda.synchronize()
    elapsed_time = start_event.elapsed_time(end_event)  # milliseconds
    return time_to_return, elapsed_time


if __name__ == '__main__':
    with open(path_to_config, "r") as f:
            config = json.load(f)
            
    code_book = config['code_book']
    quantizers = config['quantizers']
    dec_code = config['dec_code']
    enc_code = 2
    model = prepare_model(code_book = code_book, quantizers = quantizers, dec_code = dec_code, enc_code = 2)
    model.eval()
    model = model.to(device_)
    codebooks = model.quantizer.codebooks

    
    warmup_mat = read_matrix_from_file_raw(path_for_warmup)
    warmup_data = rebuild_quantized(codebooks, warmup_mat, 4)
    warm_up(model, warmup_data)
    
    data_loading = []
    inference_time = []
    runs = 10000
    for i in range(runs):
        cpu_time, gpu_time = speed_check(model, codebooks)
        data_loading.append(cpu_time)
        inference_time.append(gpu_time)
        

    average_data_loading = np.mean(data_loading)
    average_inference_time = np.mean(inference_time)
    
    print(f"Data from Indices to rebuilt Vector: {(average_data_loading) * 1000:.3f} ms over {runs} runs")
    print(f"Average forward pass: {average_inference_time:.3f} ms over {runs} runs")