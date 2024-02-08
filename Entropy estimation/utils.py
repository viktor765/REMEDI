import math
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import cmasher as cmr

import torch
from torch.utils.data import Dataset
from sklearn import datasets
from synthetic import get_random_data_generator, DataGeneratorMulti

cmap_low = matplotlib.cm.copper_r
cmap_high = cmr.get_sub_cmap('Greens', 0, 0.5)

def plot_samples(samples, n_rows=4, n_cols=5, low_end=0, high_end=1, extra_range_frac=0.25, cmap_low=cmap_low, cmap_high=cmap_high):
    side = int(np.sqrt(samples.shape[1]))
    extra_range = extra_range_frac * (high_end - low_end)

    assert samples.shape[0] == n_rows * n_cols, f"Expected {n_rows * n_cols} samples, got {samples.shape[0]}"

    samples = samples.reshape(-1, side, side)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 2, n_rows * 2))
    for i, ax in enumerate(axes.flatten()):
        if i >= samples.shape[0]:
            break

        image = samples[i]
        mask_low = samples[i] > low_end
        mask_high = samples[i] < high_end
        masked_image_low = np.ma.masked_array(samples[i], mask=mask_low)
        masked_image_high = np.ma.masked_array(samples[i], mask=mask_high)

        im = ax.imshow(image, cmap="gray", vmin=low_end, vmax=high_end, interpolation='none')
        ax.imshow(masked_image_low, cmap=cmap_low, vmin=low_end-extra_range, vmax=low_end)
        ax.imshow(masked_image_high, cmap=cmap_high, vmin=high_end, vmax=high_end+extra_range)

        ax.axis('off')

class TriangleDataset(Dataset):
    def __init__(self, n_samples, device='cpu', number=3, dim=1, seed=7):
        self.n_samples = n_samples
        self.number = number
        self.dim = dim
        self.seed = seed

        self.gen1 = get_random_data_generator('triangle', number=number, seed=seed)
        self.gen = DataGeneratorMulti(self.gen1, dim)
        self.samples = torch.tensor(self.gen.rvs((n_samples,))).float().to(device)

    def entropy(self):
        return self.gen.entropy()

    def __len__(self):
        return self.samples.shape[0]

    def __getitem__(self, idx):
        sample = self.samples[idx]
        return sample
    
class BallDataset(Dataset):
    def __init__(self, n_samples, device='cpu', dim=1, r=1.0, seed=7):
        self.n_samples = n_samples
        self.dim = dim
        self.r = r

        generator = torch.Generator().manual_seed(seed)
        self.samples = torch.randn(n_samples, dim, generator=generator)
        self.samples /= torch.norm(self.samples, dim=1).reshape(-1, 1)
        self.samples *= torch.rand(n_samples, 1, generator=generator)**(1/dim)
        self.samples *= r
        self.samples = self.samples.float().to(device)

    def volume(self):
        return np.pi**(self.dim/2) / np.math.gamma(self.dim/2 + 1) * self.r**self.dim
    
    def entropy(self):
        return np.log(self.volume())

    def cross_entropy(self, sigma2):
        '''
        Computes the cross entropy C(B, N(0, sigma2 * I_d))
        '''
        return math.log((2*math.pi*sigma2)**(self.dim/2)) + 1/(2*sigma2) * self.dim / (self.dim+2) * self.r**2
    
    def relative_entropy(self, sigma2):
        '''
        Computes the relative entropy R(B || N(0, sigma2 * I_d))
        '''
        #return math.log(math.gamma(self.dim/2 + 1) / (math.pi**(self.dim/2) * self.r**self.dim)) + math.log((2*math.pi*sigma2)**(self.dim/2)) + 1/(2*sigma2) * self.dim / (self.dim+2) * self.r**2#(1 / self.r**2)

        return -self.entropy() + self.cross_entropy(sigma2)
    
    def variance(self):
        '''
        Theoretical (marginal) variance of a sample from the ball
        '''
        return self.r**2 / (self.dim + 2)

    def emp_variance(self):
        '''
        Empirical (marginal) variance of the samples
        '''
        return self.samples.cpu().numpy().var(axis=0).mean()

    @staticmethod
    def radius_ball(d):
        '''
        Radius such that volume of ball is 1
        '''
        return (math.gamma(d/2 + 1) / math.pi**(d/2))**(1/d)

    def __len__(self):
        return self.samples.shape[0]

    def __getitem__(self, idx):
        sample = self.samples[idx]
        return sample
    
class HypercubeDataset(Dataset):
    '''
    A dataset of samples from the d-dimensional hypercube [0, side]^d
    '''

    def __init__(self, n_samples, device='cpu', dim=1, side=1.0, seed=7):
        self.n_samples = n_samples
        self.dim = dim
        self.side = side

        self.samples = torch.rand(n_samples, dim, generator=torch.Generator().manual_seed(seed))
        self.samples *= side
        self.samples = self.samples.float().to(device)

    def volume(self):
        return self.side**self.dim
    
    def entropy(self):
        return np.log(self.volume())
    
    def cross_entropy(self, sigma2):
        '''
        Computes the cross entropy C(C, N(1/2, sigma2 * I_d))
        '''
        #return math.log((2*math.pi*sigma2)**(self.dim/2)) + 1/(2*sigma2) * self.dim / (self.dim+2) * self.side**2

        return self.dim/2 * (math.log(2*math.pi*sigma2) + 1/12*self.side**2/sigma2)
    
    def relative_entropy(self, sigma2):
        '''
        Computes the relative entropy R(C || N(1/2, sigma2 * I_d))
        '''
        return -self.entropy() + self.cross_entropy(sigma2)
    
    def variance(self):
        '''
        Theoretical (marginal) variance of a sample from the hypercube
        '''
        return self.side**2 / 12
    
    def emp_variance(self):
        '''
        Empirical (marginal) variance of the samples
        '''
        return self.samples.cpu().numpy().var(axis=0).mean()

    def __len__(self):
        return self.samples.shape[0]

    def __getitem__(self, idx):
        sample = self.samples[idx]
        return sample

class MoonsDataset(Dataset):
    def __init__(self, n_samples, noise, device='cpu', seed=7):
        self.samples, _ = datasets.make_moons(n_samples=n_samples, noise=noise, random_state=seed)
        self.samples = torch.tensor(self.samples).float().to(device)

    def __len__(self):
        return self.samples.shape[0]

    def __getitem__(self, idx):
        sample = self.samples[idx]
        return sample
