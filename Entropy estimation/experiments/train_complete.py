# %%
# To be able to import from parent directory
import sys
from pathlib import Path
sys.path.append(str(Path.cwd().parent))

# %%

import numpy as np
import random

import torch
from torch.utils.data import Dataset, DataLoader
from KNIFE import KNIFE, fit_kernel
from REMEDI import REMEDI, train_model
from synthetic import get_random_data_generator, DataGeneratorMulti
from utils import TriangleDataset, MoonsDataset, BallDataset, HypercubeDataset
import argparse
import copy
import pickle
from datetime import datetime

from matplotlib import pyplot as plt

# %%

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# %%

args_data = argparse.Namespace()
args_data.type = 'triangle'
args_data.n_samples = 50000
args_data.number = 2
args_data.dim = 8
args_data.shape_seed = 7

args_KNIFE = argparse.Namespace()
args_KNIFE.batchsize = 128
args_KNIFE.num_modes = 16
args_KNIFE.epochs = 30
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
args_REMEDI.hidden_dim = [500, 200]
args_REMEDI.one_hot_y = False
args_REMEDI.output_type = "eT"
args_REMEDI.use_f_div = False
args_REMEDI.use_weight_averaged_samples = False
args_REMEDI.n_epochs = 10
args_REMEDI.batchsize = 1000
args_REMEDI.sample_size = 100
args_REMEDI.lr = 0.001
args_REMEDI.lr_base_dist = 0.001

# %%

def train_complete_model(args_data, args_REMEDI, args_KNIFE, base_dist_filename = None, seed = 7, device = 'cpu', log_parent_dir = None):
    #check first two arguments are not both None
    if base_dist_filename == None and args_KNIFE == None:
        raise ValueError('base_dist_filename and args_KNIFE cannot both be None')
    
    #check that log_parent_dir exists
    if log_parent_dir != None:
        if not log_parent_dir.exists():
            raise ValueError('log_parent_dir does not exist')
        
    assert args_data.type in ['triangle', 'two-moons', 'ball', 'cube']

    log = False
    if log_parent_dir != None:
        if log_parent_dir.exists():
            save_dir = log_parent_dir / f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}_seed{seed}"
            save_dir.mkdir(parents=True, exist_ok=True)
            log = True
        else:
            raise ValueError('log_parent_dir does not exist')
    else:
        print('log_parent_dir is None, not logging/saving')

    if log:
        with open(save_dir / 'args.txt', 'w') as f:
            f.write(str(args_data) + '\n')
            f.write(str(args_REMEDI) + '\n')
            f.write(str(args_KNIFE) + '\n')
        with open(save_dir / 'args.pkl', 'wb') as f:
            pickle.dump((args_data, args_REMEDI, args_KNIFE), f)

    if log:
        with open(save_dir / 'seed.txt', 'w') as f:
            f.write(f"Seed: {seed}\n")

    start_time = datetime.now()
            
    # seed RNGs
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)

    def worker_init_fn(worker_id):
        random.seed(seed + worker_id)
        np.random.seed(seed + worker_id)
    
    if args_data.type == 'triangle':
        train_dataset = TriangleDataset(args_data.n_samples, device, args_data.number, args_data.dim, args_data.shape_seed)
        test_dataset = TriangleDataset(args_data.n_samples, device, args_data.number, args_data.dim, args_data.shape_seed)
        train_loader = DataLoader(train_dataset, worker_init_fn=worker_init_fn, batch_size=args_KNIFE.batchsize, shuffle=True)
        test_loader = DataLoader(test_dataset, worker_init_fn=worker_init_fn, batch_size=args_KNIFE.batchsize, shuffle=True)
    elif args_data.type == 'ball':
        train_dataset = BallDataset(args_data.n_samples, device, args_data.dim, args_data.r, seed=seed)
        test_dataset = BallDataset(args_data.n_samples, device, args_data.dim, args_data.r, seed=seed+100)
        train_loader = DataLoader(train_dataset, worker_init_fn=worker_init_fn, batch_size=args_KNIFE.batchsize, shuffle=True)
        test_loader = DataLoader(test_dataset, worker_init_fn=worker_init_fn, batch_size=args_KNIFE.batchsize, shuffle=True)
    elif args_data.type == 'cube':
        train_dataset = HypercubeDataset(args_data.n_samples, device, args_data.dim, seed=seed)
        test_dataset = HypercubeDataset(args_data.n_samples, device, args_data.dim, seed=seed+100)
        train_loader = DataLoader(train_dataset, worker_init_fn=worker_init_fn, batch_size=args_KNIFE.batchsize, shuffle=True)
        test_loader = DataLoader(test_dataset, worker_init_fn=worker_init_fn, batch_size=args_KNIFE.batchsize, shuffle=True)
    elif args_data.type == 'two-moons':
        train_dataset = MoonsDataset(args_data.n_samples, 0.05, device, seed=seed)
        test_dataset = MoonsDataset(args_data.n_samples, 0.05, device, seed=seed+100)
        train_loader = DataLoader(train_dataset, worker_init_fn=worker_init_fn, batch_size=args_KNIFE.batchsize, shuffle=True)
        test_loader = DataLoader(test_dataset, worker_init_fn=worker_init_fn, batch_size=args_KNIFE.batchsize, shuffle=True)

    # if has property entropy, check that train and test dataset entropies match
    if hasattr(train_dataset, 'entropy'):
        assert train_dataset.entropy() == test_dataset.entropy(), f"Train and test dataset entropies do not match (train: {train_dataset.entropy()}, test: {test_dataset.entropy()})"

    if log:
        with open(save_dir / 'train-test_dataset.pkl', 'wb') as f:
            pickle.dump((train_dataset, test_dataset), f)

    #copy args_KNIFE to avoid changing the global args_KNIFE
    args_KNIFE = copy.deepcopy(args_KNIFE)
    args_KNIFE.train_samples = train_loader
    args_KNIFE.test_samples = test_loader

    if hasattr(train_dataset, 'entropy'):
        print(f"True entropy: {train_dataset.entropy()}")

    if hasattr(train_dataset, 'entropy') and log:
        with open(save_dir / 'true_entropy.pkl', 'wb') as f:
            pickle.dump(train_dataset.entropy(), f)

    if base_dist_filename is not None:
        base_dist0 = KNIFE.loadKNIFE(base_dist_filename)
    else:
        print("Fitting base distribution (KNIFE)")
        base_dist0, knife_train_losses, knife_val_losses = fit_kernel(args_KNIFE, True, return_history=True)
    
    if log:
        torch.save(base_dist0.state_dict(), save_dir / 'base_dist0.pt')

        #if knife_train_losses variable exists, save it
        if 'knife_train_losses' in locals():
            with open(save_dir / 'knife_train_val_losses.pkl', 'wb') as f:
                pickle.dump((knife_train_losses, knife_val_losses), f)

    if 'knife_train_losses' in locals():
        # Plot KNIFE losses
        plt.clf()
        if hasattr(train_dataset, 'entropy'):
            plt.axhline(train_dataset.entropy(), color='g', linestyle='--', label="True")
        plt.plot(knife_train_losses, label="Train")
        plt.plot(knife_val_losses, label="Validation")
        plt.legend()
        plt.xlabel("Epoch")
        plt.ylabel("Entropy")
        plt.savefig(save_dir / 'KNIFE_train_val_losses.png')
        #plt.show()

    base_dist = copy.deepcopy(base_dist0)

    mini = REMEDI(base_dist, args_REMEDI.train_base_dist, args_REMEDI.hidden_dim, args_REMEDI.one_hot_y, args_REMEDI.output_type).to(device)
    optimizer = torch.optim.Adam([
        {'params': list(mini.ln.parameters())},
        {'params': mini.gc0.parameters()},
        {'params': mini.gc1.parameters()}
    ], maximize=False, lr=args_REMEDI.lr, weight_decay=0.0001)
    if args_REMEDI.train_base_dist:
        optimizer.add_param_group({'params': mini.base_dist.parameters(), 'lr': args_REMEDI.lr_base_dist, 'weight_decay': 0})

    # Recreate data loaders, with REMEDI parameters
    train_loader = DataLoader(train_dataset, worker_init_fn=worker_init_fn, batch_size=args_REMEDI.batchsize, shuffle=True)
    test_loader = DataLoader(test_dataset, worker_init_fn=worker_init_fn, batch_size=args_REMEDI.batchsize, shuffle=True)

    # Train model
    remedi_train_losses, remedi_val_losses = train_model(mini, args_REMEDI.use_f_div, args_REMEDI.use_weight_averaged_samples, optimizer, train_loader, test_loader, args_REMEDI.n_epochs, args_REMEDI.batchsize, args_REMEDI.sample_size)

    if log:
        with open(save_dir / 'remedi_train_val_losses.pkl', 'wb') as f:
            pickle.dump((remedi_train_losses, remedi_val_losses, base_dist0.cross_entropy), f)

        torch.save(mini, save_dir / 'remedi.pt')

    # Plot REMEDI losses
    plt.clf()
    plt.axhline(base_dist0.cross_entropy, color='r', linestyle='--', label="KNIFE")
    if hasattr(train_dataset, 'entropy'):
        plt.axhline(train_dataset.entropy(), color='g', linestyle='--', label="True")
    plt.plot(remedi_train_losses, label="Train")
    plt.plot(remedi_val_losses, label="Validation")
    plt.legend()
    plt.xlabel("Epoch")
    plt.ylabel("Entropy")
    plt.savefig(save_dir / 'REMEDI_train_val_losses.png')
    #plt.show()

    end_time = datetime.now()
    total_time = end_time - start_time

    if log:
        with open(save_dir / 'time.txt', 'w') as f:
            f.write(f"Start time: {start_time}\n")
            f.write(f"End time: {end_time}\n")
            f.write(f"Total time: {total_time}\n")

    return mini, remedi_train_losses, remedi_val_losses

# %%

# Test
if __name__ == "__main__":
    log_parent_dir = Path.cwd() / 'runs' / 'train_complete_testing' / ("test-" + datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
    log_parent_dir.mkdir(parents=True, exist_ok=True)

    # Train model
    remedi, train_losses, val_losses = train_complete_model(args_data=args_data, args_REMEDI=args_REMEDI, args_KNIFE = args_KNIFE, base_dist_filename = None, device=device, log_parent_dir=log_parent_dir)

# %%
