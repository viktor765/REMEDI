from imghdr import tests
import torch
import torch.nn
import torch.nn.init
import torchvision#, torchtext
from torchvision.datasets import ImageFolder
import argparse
import sklearn.datasets
import numpy as np
import random
import os
from typing import Any
import glob

class Dataset(torch.utils.data.Dataset):

    def __init__(self,X,Y):
        self.data = X
        self.targets = Y
    
    def __getitem__(self,index):
        data, target = torch.from_numpy(self.data[index,:]).type('torch.FloatTensor')\
                        , torch.from_numpy(self.targets[index]).squeeze()
        return data, target
    
    def __len__(self):
        return len(self.targets)

class ImageNetDataset(torch.utils.data.Dataset):
    def __init__(self,data_dir):
        os.chdir(data_dir)
        X_files = glob.glob('class - *.npy')
        self.data_dir = data_dir
        self.X_files = X_files  
    
    def __getitem__(self,index):
        data = torch.from_numpy(np.load(self.data_dir + self.X_files[index]))
        target = int(self.X_files[index].split(' ')[2])
        return data,target
    
    def __len__(self):
        return len(self.X_files)

class ImageNetValDataset(torch.utils.data.Dataset):
    def __init__(self,X,y):
        self.data = torch.from_numpy(np.load(X))
        self.targets = torch.from_numpy(np.load(y))  
    
    def __getitem__(self,index):
        data = self.data[index,]
        target = self.targets[index,]
        return data,target
    
    def __len__(self):
        return len(self.targets)


def get_mnist(d=0):
    ''' 
    This function returns the MNIST dataset in training, validation, test splits.
    '''

    trainset = torchvision.datasets.MNIST(root='../data', train=True, download=True, \
        transform=torchvision.transforms.Compose([torchvision.transforms.ToTensor(), \
        torchvision.transforms.Normalize((0.0,), (1.0,))]))

    testset = torchvision.datasets.MNIST(root='../data', train=False, download=True, \
        transform=torchvision.transforms.Compose([torchvision.transforms.ToTensor(), \
        torchvision.transforms.Normalize((0.0,), (1.0,)), torchvision.transforms.RandomRotation(degrees=(d,d))]))
    
    return trainset, testset


def get_CIFAR10(d=0):
    data_path_C10 = "../data/CIFAR10/"
    mean = [0.4914, 0.4822, 0.4465]
    std = [0.2023, 0.1994, 0.2010]
    transform_train = torchvision.transforms.Compose([
        torchvision.transforms.RandomCrop(32, padding=4),
        torchvision.transforms.RandomHorizontalFlip(),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        torchvision.transforms.RandomRotation(degrees=(d,d))
    ])
    trainset = torchvision.datasets.CIFAR10(root=data_path_C10, train=True,
                    download=True, transform=transform_train)
    testset = torchvision.datasets.CIFAR10(root=data_path_C10, train=False,
                    download=True, transform=transform_test)
    return trainset,testset


def get_IMNet1():
    train_data_dir = '../data/ImageNet-ILSVRC2012T/train/'
    val_data_X = '../data/ImageNet-ILSVRC2012T/val/val_x_data.npy'
    val_data_y = '../data/ImageNet-ILSVRC2012T/val/val_y_data.npy'
    return ImageNetDataset(train_data_dir),ImageNetValDataset(val_data_X,val_data_y)


def get_data(dataset='mnist',d=0,noisy=False):
    '''
    This function returns the training and validation set from MNIST
    '''

    if dataset == 'mnist':
        return get_mnist(d)
    elif dataset == 'CIFAR10':
        return get_CIFAR10(d)
    elif dataset == 'ImageNet':
        return get_IMNet1()

def get_args():
    '''
    This function returns the arguments from terminal and set them to display
    '''

    parser = argparse.ArgumentParser(
        description = 'Run convex IB Lagrangian on MNIST dataset (with Pytorch)',
        formatter_class = argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument('--repl_n', type = int, default = 1,
        help = 'replication number')
    parser.add_argument('--n_epochs', type = int, default = 100,
        help = 'number of training epochs')
    parser.add_argument('--beta', type = float, default = 0.0,
        help = 'Lagrange multiplier (only for train_model)')
    parser.add_argument('--u_func_name', choices = ['pow', 'exp','shifted-exp','none'], default = 'exp',
        help = 'monotonically increasing, strictly convex function')
    parser.add_argument('--hyperparameter', type = float, default = 1.0,
        help = 'hyper-parameter of the h function (e.g., alpha in the power and eta in the exponential case)')
    parser.add_argument('--compression', type = float, default = 1.0,
        help = 'desired compression level (in bits). Only for the shifted exponential.')
    parser.add_argument('--K', type = int, default = 2,
        help = 'Dimensionality of the bottleneck varaible')
    parser.add_argument('--sgd_batch_size', type = int, default = 128,
        help = 'mini-batch size for the SGD on the error')
    parser.add_argument('--dataset', choices = ['mnist','CIFAR10','ImageNet'], default = 'mnist',
        help = 'dataset where to run the experiments')
    parser.add_argument('--optimizer_name', choices = ['sgd', 'rmsprop', 'adadelta', 'adagrad', 'adam', 'asgd'], default = 'adam',
        help = 'optimizer')
    parser.add_argument('--method', choices = [ 'variational_IB','IB_w_KNIFE','IB_w_REMEDI','IB_w_REMEDI_Gauss','IB_w_MINE'], default =  'variational_IB',
        help = 'information bottleneck computation method')
    parser.add_argument('--learning_rate', type = float, default = 0.0001,
        help = 'initial learning rate')
    parser.add_argument('--learning_rate_drop', type = float, default = 0.6,
        help = 'learning rate decay rate (step LR every learning_rate_steps)')
    parser.add_argument('--learning_rate_steps', type = int, default = 10,
        help = 'number of steps (epochs) before decaying the learning rate')
    parser.add_argument('--gpu', type=int, default=0, help='gpu device id')
    return parser.parse_args()
