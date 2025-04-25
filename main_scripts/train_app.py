from torch.utils.data import TensorDataset, DataLoader
from torchinfo import summary
import torch
import multiprocessing as mp 
import sys
import os 
import time
import argparse
import ast

sys.path.append("../src")

from train import Train, save_model, plot_loss
from models import AutoEncoder
from model_config import ModelConfig
import utils
from utils import AeroDataset
import yaml
import json



window_size = 800
n_channels = 36
workers_num_ = 4
local_path_train = "../mid_results/train-set/*"  # These two paths are relative and are setup in the AeroSense guide
local_path_valid = "../mid_results/valid-set/*"


save_config = ModelConfig()

def init_args():
    # options
        parser = argparse.ArgumentParser(description='Training')
        parser.add_argument('-a', '--alpha', type=float, help='alpha parameter')
        parser.add_argument('-b1', '--beta1', type=float, help='betas parameter')
        parser.add_argument('-b2', '--beta2', type=float, help='betas parameter')
        parser.add_argument('-l', '--lr', type=float, help='learning rate')
        parser.add_argument('-e', '--eps', type=float, help='eps parameter')
        parser.add_argument('-c', '--code_book', type=int, help='code_book size')
        parser.add_argument('-q', '--quantizers', type=int, help='number of quantizers')
        parser.add_argument('--dec_code', type=int, help='decoder code')
        parser.add_argument('--enc_code', type=int, help='encoder code')
        parser.add_argument('-g', '--gamma', type=float, help='gamma parameter')
        parser.add_argument('--eta', type=float, help='eta parameter')
        parser.add_argument('--disc_bool', type=bool, help='discriminator boolean')
        parser.add_argument('--epochs', type=int, help='number of epochs')
        args = parser.parse_args()
        return args

def RAM_checker ():

    # Getting all memory using os.popen()
    total_memory, used_memory, free_memory = map(int, os.popen('free -t -m').readlines()[-1].split()[1:])
    # Memory usage
    print("RAM free: " + str(free_memory) + " MB")
    print("RAM used: " + str(round((used_memory/total_memory)*100,2)) + "%")
    #return memory usage in %
    return ((used_memory/total_memory)*100)


def aero_DataLoader(device: str, dataset: torch.utils.data.Dataset, workers_num: int, batch_size: int):
    if device == "cpu":
        return  DataLoader(dataset, batch_size = batch_size, shuffle = True, pin_memory=False,num_workers = workers_num)
    return DataLoader(dataset, batch_size = batch_size, shuffle = True, pin_memory=True,num_workers = workers_num)

def save_config_to_json(config, path):
    with open(path, 'w') as json_file:
        json.dump(config, json_file, indent=4)

if __name__ == '__main__': 
    mp.set_start_method('spawn')

    args = init_args()
    
        
    save_config.alpha = 0.9
    save_config.batch_size = 128
    save_config.epochs = 101
    save_config.lr = 0.0001
    save_config.betas = (0.9, 0.98)
    save_config.eps = 4 * 1e-9
    save_config.code_book = 768 
    save_config.quantizers = 1
    save_config.RVQ = True
    save_config.disc_bool = True        # this boolean is to select to train with a discriminator, needs to be specified in the Train() call, by default False
    save_config.gamma = 1               # weight for discriminator loss
    save_config.eta = 1                 # weight for RVQ loss
    save_config.enc_code = 2            # IMPORTANT: If using enc_code 1, you will use the encoder with Batchnorm
    #                                                If using enc_code 2, you will use the encoder without Batchnorm

    save_config.dec_code = 3            # IMPORTANT: This is for the choice of decoder architecture. Cases: - = 1 is Original Decoder
    #                                                                                                       - = 2 is slightly altered decoder with channelwise layer
    #                                                                                                       - = 3 is Complex Decoder
    appendix = ""

    if (args.alpha):
        save_config.alpha=args.alpha
        appendix += "alpha:" + str(save_config.alpha)
    if (args.beta1):
        betas = (args.beta1, args.beta2)
        appendix += "betas:" + str(betas)
    if (args.lr):
        lr=args.lr
        appendix += "lr:" + str(lr)
    if (args.eps):
        eps=args.eps
        appendix += "eps:" + str(eps)
    if (args.code_book):
        code_book = args.code_book
        appendix += "code_book:" + str(code_book)
    if (args.quantizers):
        quantizers = args.quantizers
        appendix += "quantizers:" + str(quantizers)

    
    run_name = "PRELU" + appendix
   
    path_to_save_losses = "../mid_results/losses_csv"

    unique_run_name = run_name

    path_to_save_model = "../mid_results/models/" + unique_run_name
    if (not os.path.exists(path_to_save_model)):
        os.mkdir(path_to_save_model)
    
    # Save the configuration to a JSON file
    config_json_path = os.path.join(path_to_save_model, "config.json")
    save_config_to_json(save_config.__dict__, config_json_path)


    arch_id = "ss00"


    model = AutoEncoder(arch_id = arch_id, c_in = n_channels, RVQ = save_config.RVQ, 
                    codebook_size=save_config.code_book, quantizers=save_config.quantizers, 
                    dec_code=save_config.dec_code, enc_code=save_config.enc_code)
    
    summary(model, input_size = (1, 36, window_size), verbose = 1, depth = 5)
    
    
    #train metrics
    tot_train_time = []
    tot_train_loss = []
    tot_valid_loss = []

    device_ = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    aero_dataset_train = AeroDataset(local_path_train, device = 'cpu')
    aero_dataset_valid = AeroDataset(local_path_valid, device = 'cpu')
    
    aero_dl_train = aero_DataLoader(device= device_, 
                                    dataset= aero_dataset_train, 
                                    workers_num=workers_num_, 
                                    batch_size=save_config.batch_size)
    
    aero_dl_valid = aero_DataLoader(device= device_, 
                                    dataset= aero_dataset_valid, 
                                    workers_num=workers_num_, 
                                    batch_size=save_config.batch_size)
    

    train_time, train_loss, valid_loss = Train(model = model, epochs = save_config.epochs, 
                                            alpha = save_config.alpha, output_filter = False, 
                                            path = path_to_save_model, early_stop = False, disc_bool = save_config.disc_bool,
                                            run_name=run_name, lr=save_config.lr, gamma = save_config.gamma, eta = save_config.eta,
                                            betas=save_config.betas, eps=save_config.eps, track=True)(aero_dl_train, aero_dl_valid)
    
    tot_train_time.append(train_time)
    tot_train_loss.append(train_loss)
    tot_valid_loss.append(valid_loss)



    model_id =  save_model(model, path_to_save_model, "ss00", tot_train_loss, tot_valid_loss, save_np = True)
    plot_loss(tot_train_loss, tot_valid_loss, model_id, path_to_save_model)
 
    save_config.arch_id = arch_id
    save_config.model_id = model_id
    save_config.latent_channels = n_channels 
    save_config.train_loss = min(tot_train_loss[-1])
    save_config.valid_loss = min(tot_valid_loss[-1])
    save_config.window_size = window_size
    save_config.wind_speed = 30
     
    save_config.description = f"lp [0-5Hz] + biased values + padding_mode='replicate'."

    save_config_dict = save_config.__dict__
    utils.write_to_csv(save_config_dict)
    print(save_config_dict)

    print(f"Training time: {(tot_train_time)}")   
    print(f"Training Loss: {tot_train_loss}")
    print(f"Validation Loss: {tot_valid_loss}")
