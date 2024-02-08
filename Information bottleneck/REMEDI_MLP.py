# %%
import numpy as np

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from KNIFE import KNIFE
from sklearn import datasets
import torch.nn.init as init
from matplotlib import pyplot as plt

class REMEDI(nn.Module):
    def __init__(self, base_dist: KNIFE, train_base_dist: bool, hidden_dim = [100, 100], one_hot_y=False):
        super().__init__()

        #self.base_dist = base_dist.requires_grad_(train_base_dist)

        self.one_hot_y = one_hot_y

        #self.gc0 = nn.Conv1d(in_channels=base_dist.K, out_channels=base_dist.d*base_dist.K, kernel_size=base_dist.d, groups=base_dist.K, bias=False)

        self.ln = nn.ModuleList()
        self.ln.append(nn.Linear(base_dist.d, hidden_dim[0]))

        for i in range(1, len(hidden_dim)):
            self.ln.append(nn.Linear(hidden_dim[i-1], hidden_dim[i]))

        #self.gc1 = nn.Conv1d(in_channels=base_dist.K, out_channels=base_dist.K, kernel_size=hidden_dim[-1], groups=base_dist.K, bias=True)

    def forward(self, x):    
        #Compute neural network output
        for i in range(len(self.ln)):
            if i == range(len(self.ln))[-1]:
                x = self.ln[i](x)
            else:
                x = self.ln[i](x)
                x = torch.relu(x)
        return x# + self.base_dist.log_prob(x)
