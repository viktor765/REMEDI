# %%
import numpy as np

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from KNIFE import KNIFE
from sklearn import datasets

from matplotlib import pyplot as plt

class REMEDI(nn.Module):
    def __init__(self, base_dist: KNIFE, train_base_dist: bool, hidden_dim = [100, 100], one_hot_y = False, output_type = "T", eps: float = 1e-8):
        super().__init__()

        assert output_type in ["T", "eT"], "output_type must be either 'T' or 'eT'"

        self.base_dist = base_dist.requires_grad_(train_base_dist)

        self.output_type = output_type
        self.eps = eps

        self.one_hot_y = one_hot_y

        self.gc0 = nn.Conv1d(in_channels=base_dist.K, out_channels=base_dist.d*base_dist.K, kernel_size=base_dist.d, groups=base_dist.K, bias=False)

        self.ln = nn.ModuleList()
        self.ln.append(nn.Linear(base_dist.d, hidden_dim[0]))

        for i in range(1, len(hidden_dim)):
            self.ln.append(nn.Linear(hidden_dim[i-1], hidden_dim[i]))

        self.gc1 = nn.Conv1d(in_channels=base_dist.K, out_channels=base_dist.K, kernel_size=hidden_dim[-1], groups=base_dist.K, bias=True)

    def forward(self, x):    
        weights = torch.softmax(self.base_dist.weigh, dim=1).squeeze(0)
        var = self.base_dist.logvar.exp()
        tri = torch.tril(self.base_dist.tri, diagonal=-1)

        V = torch.diag_embed(var)
        L = V + tri @ V#Cholesky factor of precision matrix
    
        offsets = x.unsqueeze(1) - self.base_dist.means
        offsets = (L @ offsets.unsqueeze(-1)).squeeze(-1)
        y = torch.sum(offsets ** 2, dim=2)
        y = -y / 2 + torch.sum(torch.log(torch.abs(var) + 1e-8), dim=-1)# + w
        y = weights * torch.exp(y)
        #normalize y
        y = y / torch.sum(y, dim=1).unsqueeze(-1)

        if self.one_hot_y:
            y_new = torch.zeros_like(y)
            y_new[torch.arange(y.shape[0]), torch.argmax(y, dim=1)] = 1
            y = y_new

        #Compute neural network output
        x = offsets
        x = self.gc0(x).reshape(x.shape)
        for i in range(len(self.ln)):
            x = self.ln[i](x)
            x = torch.relu(x)

        x = self.gc1(x)

        #dot product y with x
        x = torch.sum(x.squeeze(-1) * y, dim=1)

        if self.output_type == "T":
            return x
        elif self.output_type == "eT":
            return nn.functional.elu(x, 1) + 1 + self.eps
        
class MINI_CNN_MNIST(nn.Module):
    def __init__(self, base_dist: KNIFE, train_base_dist: bool, output_type: str = "T", side_length: int = 28, eps: float = 1e-8):
        super().__init__()

        assert output_type in ["T", "eT"], "output_type must be either 'T' or 'eT'"
        assert side_length in [28, 14], "side_length must be either 28 or 14"

        self.base_dist = base_dist.requires_grad_(train_base_dist)
        
        self.output_type = output_type
        self.side_length = side_length
        self.eps = eps

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=5, stride=1, padding=2, groups=1, bias=True)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, stride=1, padding=2, groups=1, bias=True)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=5, stride=1, padding=2, groups=1, bias=True)
        self.fc1 = nn.Linear(7*7*128, 1000)
        self.fc2 = nn.Linear(1000, 1)

    def forward(self, x):
        x = x.reshape(x.shape[0], 1, self.side_length, self.side_length)

        x = self.conv1(x)
        x = torch.relu(x)

        if self.side_length == 28:
            x = torch.nn.functional.max_pool2d(x, 2)
            # format will be now be [batch_size, 32, 14, 14]

        x = self.conv2(x)
        x = torch.relu(x)
        x = nn.functional.max_pool2d(x, 2)

        x = self.conv3(x)
        x = torch.relu(x)

        x = x.view(-1, 7*7*128)
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.fc2(x)

        if self.output_type == "T":
            return x
        
        elif self.output_type == "eT":
            return nn.functional.elu(x, 1) + 1 + self.eps

def train_model(model: REMEDI, use_f_div: bool, use_weight_averaged_samples: bool, optimizer, train_loader, test_loader, n_epochs, batchsize, sample_size, clip_grad_norm = None, verbose: bool = False):
    # Create list to store losses
    train_losses = []
    train_comp0 = []
    train_comp1 = []
    train_comp2 = []
    val_losses = []
    val_comp0 = []
    val_comp1 = []
    val_comp2 = []

    for epoch in range(n_epochs):
        train_loss = 0
        train_comp0_epoch = 0
        train_comp1_epoch = 0
        train_comp2_epoch = 0
        for i, sample in enumerate(train_loader):
            # Dirty fix for MNIST
            if type(sample) == list:
                sample = sample[0]            

            comp0 = -model.base_dist.log_prob(sample.float()).mean()

            output1 = model(sample.float())
            if model.output_type == "eT":
                output1 = output1.log()

            comp1 = torch.mean(output1)

            if not use_weight_averaged_samples:
                # Sample from base distribution
                z = model.base_dist.sample((batchsize,)).float()
                output2 = model(z)
                if model.output_type == "T":
                    output2 = output2.exp()

                if use_f_div:
                    comp2 = (torch.e ** -1) * torch.mean(output2)
                else:
                    comp2 = torch.log(torch.mean(output2))
            else:
                if model.output_type == "T":
                    weighting_lambda = lambda z: torch.exp(model(z))
                elif model.output_type == "eT":
                    weighting_lambda = lambda z: model(z)

                weight_averaged_sample = model.base_dist.weight_averaged_sample_function(weighting_lambda, sample_size)

                if use_f_div:
                    comp2 = (torch.e ** -1) * torch.mean(weight_averaged_sample)
                else:
                    comp2 = torch.log(torch.mean(weight_averaged_sample))

            loss = comp0 - (comp1 - comp2)

            # Backpropagate
            loss.backward()

            if clip_grad_norm is not None:
                grad_norm_unclipped = torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad_norm, norm_type=2)

            if verbose:    
                print()
                print(f"Batch loss: {loss.item():.2f}, comp0: {comp0.item():.2f}, comp1: {comp1.item():.2f}, comp2: {comp2.item():.2f} on batch {i} of epoch {epoch}")
                if clip_grad_norm is not None:
                    print(f"Gradient (2-)norm unclipped: {grad_norm_unclipped.item():.2f}" + f", clipped to {clip_grad_norm}" if grad_norm_unclipped > clip_grad_norm else "")
                print()

            # Update parameters
            optimizer.step()

            # Reset gradients
            optimizer.zero_grad()

            # Add loss to epoch loss
            train_loss += loss.item()
            train_comp0_epoch += comp0.item()
            train_comp1_epoch += comp1.item()
            train_comp2_epoch += comp2.item()

        #validate
        model.eval()

        val_loss = 0
        val_comp0_epoch = 0
        val_comp1_epoch = 0
        val_comp2_epoch = 0
        with torch.no_grad():
            for i_test, sample in enumerate(test_loader):
                # Dirty fix for MNIST
                if type(sample) == list:
                    sample = sample[0]  

                comp0 = -model.base_dist.log_prob(sample.float()).mean()

                output1 = model(sample.float())
                if model.output_type == "eT":
                    output1 = output1.log()

                comp1 = torch.mean(output1)

                #if not use_weight_averaged_samples:
                
                # Sample from base distribution
                z = model.base_dist.sample((batchsize,)).float()
                output2 = model(z)
                if model.output_type == "T":
                    output2 = output2.exp()

                if use_f_div:
                    comp2 = (torch.e ** -1) * torch.mean(output2)
                else:
                    comp2 = torch.log(torch.mean(output2))

                #Not using weight averaged samples

                loss = comp0 - (comp1 - comp2)

                # Add loss to epoch loss
                val_loss += loss.item()
                val_comp0_epoch += comp0.item()
                val_comp1_epoch += comp1.item()
                val_comp2_epoch += comp2.item()

        model.train()

        # Add epoch loss to list
        train_losses.append(train_loss / (i+1))
        train_comp0.append(train_comp0_epoch / (i+1))
        train_comp1.append(train_comp1_epoch / (i+1))
        train_comp2.append(train_comp2_epoch / (i+1))
        val_losses.append(val_loss / (i_test+1))
        val_comp0.append(val_comp0_epoch / (i_test+1))
        val_comp1.append(val_comp1_epoch / (i_test+1))
        val_comp2.append(val_comp2_epoch / (i_test+1))
        print(f"Epoch {epoch+1}/{n_epochs} - Train loss: {train_losses[-1]:.3f} - Validation loss: {val_losses[-1]:.3f}")
        print(f"Train comp0: {train_comp0[-1]:.3f}, comp1: {train_comp1[-1]:.3f}, comp2: {train_comp2[-1]:.3f}")
        print(f"Val   comp0: {val_comp0[-1]:.3f}, comp1: {val_comp1[-1]:.3f}, comp2: {val_comp2[-1]:.3f}")

    return train_losses, val_losses
