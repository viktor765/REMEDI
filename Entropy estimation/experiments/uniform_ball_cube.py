# %%
# To be able to import from parent directory
import sys
from pathlib import Path
sys.path.append(str(Path.cwd().parent))

# %%

import math
import numpy as np

import torch
from torch.utils.data import Dataset, DataLoader
from KNIFE import KNIFE, fit_kernel
from REMEDI import REMEDI, train_model
from synthetic import get_random_data_generator, DataGeneratorMulti
from utils import TriangleDataset, MoonsDataset, BallDataset
from train_complete import train_complete_model
from post_plots import plot_losses, plot_triangle_1d
import argparse
import copy
import pathlib
import pickle
from datetime import datetime

from matplotlib import pyplot as plt

# %%

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# print device
print(f'Using device: {device}')

run_seeds = [0]#[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

log_parent_parent_dir = Path.cwd() / 'runs' / 'final' / datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

# parse arguments
parser = argparse.ArgumentParser(description='Run experiments')
parser.add_argument('--experiment', type=str, default='all', help='type of experiment', choices=['all', 'triangle1d', 'triangle8d', 'triangle20d', 'two-moons', 'ball10d', 'cube10d'])
parser.add_argument('--repetitions', type=int, default=10, help='number of repetitions')

# detect if running in jupyter notebook, and if so, use default args
if 'ipykernel' in sys.modules:
    args = parser.parse_args(args=[])
else:
    args = parser.parse_args()

# %% [markdown]

# ## Ball

# %%

args_data = argparse.Namespace()
args_data.type = 'ball'
args_data.n_samples = 50000
args_data.dim = 20
args_data.r = BallDataset.radius_ball(args_data.dim)

args_KNIFE = argparse.Namespace()
args_KNIFE.batchsize = 128
args_KNIFE.num_modes = 1
args_KNIFE.epochs = 10#10#50
args_KNIFE.lr = 1e-3
args_KNIFE.shuffle = True
args_KNIFE.cov_diagonal = 'var'
args_KNIFE.cov_off_diagonal = 'var'
args_KNIFE.average = 'var'
args_KNIFE.use_tanh = False
args_KNIFE.device = device
args_KNIFE.dimension = args_data.dim

args_REMEDI = argparse.Namespace()
args_REMEDI.train_base_dist = False
args_REMEDI.hidden_dim = [200, 200]
args_REMEDI.one_hot_y = False
args_REMEDI.output_type = "T"
args_REMEDI.use_f_div = False
args_REMEDI.use_weight_averaged_samples = False
args_REMEDI.n_epochs = 100
args_REMEDI.batchsize = 1000
args_REMEDI.sample_size = 100
args_REMEDI.lr = 0.001
args_REMEDI.lr_base_dist = 0.001

repetitions = args.repetitions

if args.experiment == 'all' or args.experiment == 'triangle8d':
    print(args_data)
    print(args_KNIFE)
    print(args_REMEDI)

    log_parent_dir = log_parent_parent_dir / f"{args_data.type}_{args_data.dim}d"
    log_parent_dir.mkdir(parents=True, exist_ok=True)

    # Train model
    for seed in run_seeds[:repetitions]:
        print(f"Running seed {seed}")
        remedi, train_losses, val_losses = train_complete_model(args_data=args_data, args_REMEDI=args_REMEDI, args_KNIFE = args_KNIFE, base_dist_filename = None, seed=seed, device=device, log_parent_dir=log_parent_dir)

    # %%

    # first subdirectory of log_parent_dir
    log_dir = next(log_parent_dir.iterdir())

    fig, ax = plot_losses(log_dir)
    plt.show()

# %%

# %% [markdown]

# ## Hypercube

# %%

args_data = argparse.Namespace()
args_data.type = 'cube'
args_data.n_samples = 50000
args_data.dim = 20
args_data.side = 1.0

args_KNIFE = argparse.Namespace()
args_KNIFE.batchsize = 128
args_KNIFE.num_modes = 1
args_KNIFE.epochs = 10
args_KNIFE.lr = 1e-3
args_KNIFE.shuffle = True
args_KNIFE.cov_diagonal = 'var'
args_KNIFE.cov_off_diagonal = 'var'
args_KNIFE.average = 'var'
args_KNIFE.use_tanh = False
args_KNIFE.device = device
args_KNIFE.dimension = args_data.dim

args_REMEDI = argparse.Namespace()
args_REMEDI.train_base_dist = False
args_REMEDI.hidden_dim = [1000, 1000, 500]
args_REMEDI.one_hot_y = False
args_REMEDI.output_type = "eT"
args_REMEDI.use_f_div = False
args_REMEDI.use_weight_averaged_samples = False
args_REMEDI.n_epochs = 100
args_REMEDI.batchsize = 1000
args_REMEDI.sample_size = 100
args_REMEDI.lr = 0.0001
args_REMEDI.lr_base_dist = 0.0001

repetitions = args.repetitions

if args.experiment == 'all' or args.experiment == 'triangle8d':
    print(args_data)
    print(args_KNIFE)
    print(args_REMEDI)

    log_parent_dir = log_parent_parent_dir / f"{args_data.type}_{args_data.dim}d"
    log_parent_dir.mkdir(parents=True, exist_ok=True)

    # Train model
    for seed in run_seeds[:repetitions]:
        print(f"Running seed {seed}")
        remedi, train_losses, val_losses = train_complete_model(args_data=args_data, args_REMEDI=args_REMEDI, args_KNIFE = args_KNIFE, base_dist_filename = None, seed=seed, device=device, log_parent_dir=log_parent_dir)

    # %%

    # first subdirectory of log_parent_dir
    log_dir = sorted(log_parent_dir.iterdir())[-1]

    fig, ax = plot_losses(log_dir)
    plt.show()

# %%

