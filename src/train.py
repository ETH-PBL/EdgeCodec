# This script contains the basic training backends for the data compression of AeroSense Surface pressure data.
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg') 
import numpy as np 
import shutup 
import seaborn as sns
import time 
import tracemalloc
import copy
import torch 
import torch.nn as nn 
import torch.optim as optim
from torchinfo import summary
from tqdm import tqdm
from model_config import ModelConfig

from models import AutoEncoder, Discriminator
from criterion import ReconstructionLossMixed
import utils
import os
import csv
shutup.please()


PLOT  = True
train_config = ModelConfig()

device = "cuda" if torch.cuda.is_available() else "cpu" 

class EarlyStopper:
    def __init__(self, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = float('inf')

    def early_stop(self, validation_loss):
        err = abs((validation_loss - self.min_validation_loss))
        self.min_validation_loss = validation_loss

        if err >= self.min_delta:
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False

def plot_double(ax, x_org, x_rec, sensor, label):
    
    ax.plot(x_org[0, sensor, :].cpu().detach().numpy(), label = 'original data', color = "green")
    ax.plot(x_rec[0, sensor, :].cpu().detach().numpy(), label = label, color = "black")
    ax.set_xlabel("Time [ms]")
    ax.set_ylabel("Cp data")
    ax.set_title(f"Sensor {sensor} | Reconstructed vs original data")

    ax2 = ax.twinx()  # instantiate a second axes that shares the same x-axis
    color = 'tab:blue'
    ax2.set_ylabel("Cp Error", color=color)  # 50% transparent
    ax2.plot(np.subtract(x_org[0, sensor, :].cpu().detach().numpy(),
                         x_rec[0, sensor, :].cpu().detach().numpy()), 
                         color=color, linestyle="dashed", alpha=0.4)

    ax.legend()

def plot_original_vs_reconstructed(x_org: torch.tensor, x_rec: torch.tensor, sensor: list, label:str, path:str ) -> None:
    with sns.plotting_context("poster"):
        fig, ax = plt.subplots(2, 1, figsize=(20, 10))
        plot_double(ax[0], x_org, x_rec, sensor[0], label)    
        plot_double(ax[1], x_org, x_rec, sensor[1], label)
        plt.tight_layout()

        filename = "ephoc_" + str(label) + "_original_vs_reconstructed.png"
        plt.savefig(os.path.join(path, filename), bbox_inches='tight')

        plt.show(block=False)
        plt.pause(3)
        plt.close()



class Train(nn.Module):

    def __init__(self, run_name:str, model: object, epochs: int, alpha:float, gamma:float, eta:float,
                 output_filter, path:str, early_stop: bool = True, disc_bool: bool = False,
                 pre_trained_path: str = "",lr=0.0001, betas=(0.9, 0.98), eps=1e-8, track=True, 
                 *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model = model.to(device)
        self.epochs = epochs
        self.alpha = alpha
        self.gamma = gamma
        self.eta = eta
        self.output_filter = output_filter
        self.path = path
        self.e_stop = early_stop
        self.run_name = run_name
        self.track = track
        self.disc_bool = disc_bool
        self.best_valid = float('inf')
        # adding discriminator
        if self.disc_bool:
            print('-' * 50)
            print("training with Discriminator")
            self.discriminator = Discriminator().to(device)
            self.disc_opti = torch.optim.Adam(self.discriminator.parameters(), lr=lr, betas=betas, eps=eps) # using same coefficients like model for now
            self.adversarial_loss = nn.BCEWithLogitsLoss()


        if self.e_stop == True:
            self.early_stopper = EarlyStopper(patience=3, min_delta=5)

        print("-"*50)
        print("Setup Training...")
        criterion_method = ("mse", "smooth_l1")    
        reduction_type = "sum"
        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=betas, eps=eps)
        self.criterion = ReconstructionLossMixed(criterion_method, reduction_type, alpha = self.alpha)

        if pre_trained_path != "":
            #pre-trained model
            self.load_checkpoint(pre_trained_path)
            self.model.train()


        print(f"Used Device: {device}")
        print(f"Optimizer | lr: {lr} | betas: {betas} | eps: {eps}")
        print(f"mixed loss weight: {self.alpha} | discriminator loss weight: {self.gamma} | RVQ loss weight: {self.eta}")
        print("\n")

    def save_checkpoint(self, epoch, loss):
    
        model_number = utils.generate_hexadecimal()
        torch.save({
                'epoch': epoch,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'loss': loss,
                }, f"{self.path}{epoch}_checkpoint.pt")

        return model_number 
    
    def train_discriminator(self, x_batch, y_train):
        x_batch = x_batch.to(device)
        y_train = y_train.to(device)

        # predict actual input
        inp_pred = self.discriminator(x_batch)
        inp_labels = torch.ones_like(inp_pred)

        # predict output
        outp_pred = self.discriminator(y_train.detach())
        oupt_labels = torch.zeros_like(outp_pred)

        disc_loss = self.adversarial_loss(inp_pred, inp_labels) + self.adversarial_loss(outp_pred, oupt_labels)

        self.disc_opti.zero_grad()
        disc_loss.backward()
        self.disc_opti.step()
    
    def load_checkpoint(self, path: str):

        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        loss = checkpoint['loss']

        print("\n")
        print("Model loaded from pre-trained, Epoch: " + str(epoch) + " Loss: " + str(loss))
        print("\n")

        return epoch, loss
   
    def forward_propagation(self, x_batch, y_batch, idx, epoch):
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)
        if self.model.RVQ == True:
            y_train, self.indices, self.commit_loss = self.model.forward(x_batch.float())
            self.commit_loss = self.commit_loss



        else:
            y_train = self.model.forward(x_batch.float())
        if self.output_filter:
            y_train  = self.lp(y_train)
        train_loss = self.criterion(y_train.float(), y_batch.float())

        # disc loss
        if self.disc_bool:
            fake_preds = self.discriminator.forward(y_train)
            real_labels = torch.ones_like(fake_preds)
            self.adv_loss = self.adversarial_loss(fake_preds, real_labels)
            self.adv_loss = self.adv_loss

            train_loss = train_loss + self.gamma * torch.mean(self.adv_loss)


        if self.model.RVQ == True:
            train_loss = train_loss + (self.eta * torch.mean(self.commit_loss))

        if idx == 1 and epoch%5 == 0 and PLOT == 1:       
            plot_original_vs_reconstructed(y_batch, y_train, sensor = [5, 15], label = f"Epoch {epoch} | train", path = self.path)
        return train_loss

    def back_propagation(self, train_loss):
    
        # Backpropagation
        self.optimizer.zero_grad()
        train_loss.backward()
        self.optimizer.step()

        


    def evaluate(self, x_batch, y_batch, idx, epoch):
        
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)

        with torch.no_grad():
            self.model.eval()

            if self.model.RVQ == True:
                y_valid, indices, commit_loss = self.model.forward(x_batch.float())
            else:
                y_valid = self.model.forward(x_batch.float())

            if self.output_filter:
                y_valid = self.lp(y_valid)
            

            if idx == 1 and epoch%5 == 0 and PLOT == 1:
                plot_original_vs_reconstructed(y_batch, y_valid, sensor = [5, 15], label = f"Epoch {epoch} | validation", path = self.path)
            valid_loss = self.criterion(y_valid.float(), y_batch.float())

            mean_diff, max_diff = 0, 0
            if(epoch == self.epochs - 1):
                mean_diff = np.mean(np.absolute(np.subtract(y_batch[:, :, 100:700].cpu().detach().numpy(),
                         y_valid[:, :, 100:700].cpu().detach().numpy())))
                max_diff = np.max(np.absolute(np.subtract(y_batch[:, :, 100:700].cpu().detach().numpy(),
                         y_valid[:, :, 100:700].cpu().detach().numpy())))

            return valid_loss, mean_diff, max_diff


    def forward(self, train_x, valid_x):
        
        train_epoch_loss, valid_epoch_loss = [], []
        train_batch_loss, valid_batch_loss = [], []
        train_total_loss, valid_total_loss = [], []
        time_start = time.time()
        print("-"*50)
        print("Starting Training...")

        
        mean_diff_arr, max_diff_arr = [], []
        for epoch in np.arange(0, self.epochs):
            print(f"Epoch: {epoch+1}/{self.epochs}")
            ### TRAINING PHASE ###
            idx = 0
            for x_batch, y_batch in (train_x):
                idx += 1
                
                if self.disc_bool:
                    # Step 1: Train the Discriminator
                    self.model.eval()
                    self.discriminator.train()

                    y_train = self.model.forward(x_batch.to(device).float()) # this should work since we are in eval mode
                    self.train_discriminator(x_batch, y_train[0])

                    # Step 2: Train the Generator / AutoEncoder
                    self.model.train()

                train_loss = self.forward_propagation(x_batch, y_batch, idx = idx, epoch = epoch)
               

                train_batch_loss += [train_loss]
                train_epoch_loss += [train_loss.item()]
                self.back_propagation(train_loss)
            idx = 0 
            ### VALIDATION PHASE ###
            for x_batch, y_batch in (valid_x):
                    idx += 1 
                    valid_loss, mean_diff, max_diff = self.evaluate(x_batch, y_batch, idx = idx, epoch = epoch)
                    if (epoch == self.epochs - 1):
                        mean_diff_arr += [mean_diff]
                        max_diff_arr += [max_diff]
                    valid_batch_loss += [valid_loss]
                    valid_epoch_loss += [(valid_loss.item())]

                    # path to save best model
                    if valid_loss < self.best_valid:
                        self.best_valid = valid_loss
                        save_model(self.model, self.path,"best_model", train_total_loss, valid_total_loss, QAT = self.QAT)

            print(f"\t Train loss = {sum(train_epoch_loss)/len(train_epoch_loss):.05}, \
                    Validation Loss = {sum(valid_epoch_loss)/len(valid_epoch_loss):.05}")
            if self.model.RVQ == True:
                print("RVQ "
                + f"cmt loss: {torch.mean(self.commit_loss).item():.3f} | "
                + f"active %: {self.indices.unique().numel() / self.model.codebook_size * 100:.3f}"
                )

            train_total_loss.append(sum(train_epoch_loss)/len(train_epoch_loss))
            valid_total_loss.append(sum(valid_epoch_loss)/len(valid_epoch_loss))
            # wandb logging
            #save checkpoint
            self.save_checkpoint(epoch, sum(train_epoch_loss)/len(train_epoch_loss))

            #early stop condition
            if self.e_stop == True and self.early_stopper.early_stop(sum(train_epoch_loss)/len(train_epoch_loss)):
                break

            file_path = '../mid_results/losses.csv'
            # Define the data to be added
            if (epoch == self.epochs - 1 and self.track):
                row = {'run_name': self.run_name, 'mean_val':sum(valid_epoch_loss)/len(valid_epoch_loss), 
                'max_val':max(valid_epoch_loss), 'mean_train':sum(train_epoch_loss)/len(train_epoch_loss), 
                'max_train':max(train_epoch_loss), 
                'mean_diff': np.mean(mean_diff_arr),
                'max_diff': np.max(max_diff_arr)}
                with open(file_path, mode='a', newline='') as file:
                    writer = csv.DictWriter(file, fieldnames=row.keys())
                    
                    # Write the new row
                    writer.writerow(row)
                # torch.save(model.encoder.state_dict, "../mid_results/models/encoder_model")
                    
            train_epoch_loss, valid_epoch_loss = [], []
            train_batch_loss, valid_batch_loss = [], []
            idx = 0

        time_end = time.time()
        train_time = time_end - time_start
        return train_time, train_total_loss, valid_total_loss
    



def save_model(model, path, arch_id, train_loss, valid_loss, save_np = False, QAT = False):
    
    # Ensure the directory exists
    if not os.path.exists(path):
        os.makedirs(path)

    # Construct the full file path
    model_dict_file_path = os.path.join(path, f"{arch_id}.pt")
    model_file_path = os.path.join(path, f"{arch_id}.pth")
    model.eval()
    torch.save(model.state_dict(), model_dict_file_path)
    #torch.save(model, model_file_path)
    if save_np:
        np.save(f"{model_dict_file_path}_train_loss.npy", np.array(train_loss))
        np.save(f"{model_dict_file_path}_valid_loss.npy", np.array(valid_loss))
    print(f"Saved the model: {arch_id} with architecture Id ss00")
    model.to(torch.device(device))
    model.train()
    return arch_id 

def save_training_results(train_config):
    
    data = train_config.__dict__
    utils.write_to_csv(data)
    print("Saved the training results to the csv file.")




def plot_loss(train_loss, valid_loss, model_id, path:str):
    
    with sns.plotting_context('poster'): 
        fig, ax = plt.subplots(1, 1, figsize=(20, 10))
        ax.plot(train_loss, color = 'green', label = 'Train ')
        ax.plot(valid_loss, color = 'red', label = 'Valid ')

        ax.set_xlabel('epochs')
        ax.set_ylabel('Loss')

        ax.set_title(f"Loss trend for Model {model_id}")
        ax.legend()
        plt.tight_layout()

        filename = "plot_loss.png"
        plt.savefig(os.path.join(path, filename), bbox_inches='tight')

        plt.show(block=False)
        plt.pause(3)
        plt.close()

