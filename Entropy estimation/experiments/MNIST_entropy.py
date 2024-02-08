# %%

# To be able to import from parent directory
import sys
from pathlib import Path

# Assumes that the current working directory is Entropy-2023/experiments
sys.path.append(str(Path.cwd().parent))
print(f"{Path.cwd()=}")
print(f"{sys.path=}")

# %%

import numpy as np
import torch
from torchvision import datasets as tvdatasets, transforms
from sklearn import datasets
import matplotlib
from matplotlib import pyplot as plt
import cmasher as cmr
from datetime import datetime
import pickle

from KNIFE import fit_kernel, KNIFE
from REMEDI import REMEDI, train_model
from REMEDI import MINI_CNN_MNIST

from torch.utils.data import Dataset, DataLoader
import argparse

import logging
import os

from sklearn.mixture import GaussianMixture

# %%

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

parser = argparse.ArgumentParser(description='MNIST sklearn')
parser.add_argument('--factor', type=int, default=1, help='Factor to downsample MNIST images by')#Should be inferred from loaded model?
parser.add_argument('--noise_scale', type=float, default=1.0, help='Standard deviation of Gaussian noise added to MNIST images')#0.3081 for old behavior
parser.add_argument('--batchsize', type=int, default=200, help='Batch size')
parser.add_argument('--seed', type=int, default=8, help='Random seed')

# detect if running in jupyter notebook, and if so, use default args
if 'ipykernel' in sys.modules:
    args = parser.parse_args(args=[])
else:
    args = parser.parse_args()

print(f"{args=}")

# args.batchsize = 200
# args.num_modes = 1
# args.epochs = 50
# args.lr = 2e-3
# args.shuffle = True
# args.cov_diagonal = 'var'
# args.cov_off_diagonal = 'var'
# args.average = 'var'
# args.use_tanh = False
# args.device = device

# %%

# specify any temporary settings (for running in notebook)
#args.batchsize = 300

# %%
save_dir = Path.cwd() / 'runs' / 'MNIST_entropy' / datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
save_dir.mkdir(parents=True, exist_ok=True)

# Save args
with open(save_dir / 'args.txt', 'w') as f:
    f.write(str(args))

# Create log file
log_file = save_dir / 'log.txt'
log_file.touch(exist_ok=True)

targets = logging.StreamHandler(sys.stdout), logging.FileHandler(log_file)
logging.basicConfig(format='%(message)s', level=logging.INFO, handlers=targets)
print = logging.info
print('test')

# run command to copy this file to the save directory
os.system(f'cp {Path(__file__)} {save_dir / "script.py"}')
# get slurm job id
slurm_job_id = os.popen('echo -n $SLURM_JOB_ID').read()

if slurm_job_id == '':
    slurm_job_id = None
    
print(f"SLURM job id: {slurm_job_id}")

# %%

# seed RNGs

seed = args.seed
np.random.seed(seed)
torch.manual_seed(seed)

# %%

# Get MNIST dataloader

factor = 1
side = 28 // factor
assert side * factor == 28

noise_scale = args.noise_scale

#TODO: change 256 to 255 everywhere
mean_normalization = 0.1307
std_normalization = 0.3081
low_end = (0 - mean_normalization) / std_normalization
high_end = (1 - mean_normalization) / std_normalization

transform=transforms.Compose([
    transforms.ToTensor(),
    #Not used yet, could be added but would change the entropy too much, must then be added in test as well
    #transforms.RandomAffine(degrees=5, translate=(3/side, 1/side), interpolation=transforms.InterpolationMode.BILINEAR),#put first to avoid quantization errors
    #dequantize
    transforms.Lambda(lambda x: x + (torch.rand_like(x) - 0.5) / 255),
    #add small gaussian noise
    transforms.Lambda(lambda x: x + noise_scale * torch.randn_like(x) / 255),
    #Normalize
    transforms.Normalize((mean_normalization,), (std_normalization,)),
    #small random translation, using roll, not used here
    #transforms.Lambda(lambda x: torch.roll(x, shifts=(torch.randint(-2, 3, (1,)).item(), torch.randint(-1, 2, (1,)).item()), dims=(1, 2))),
    #convert to e.g. 14x14 by taking mean of 2x2 patches
    transforms.Lambda(lambda x: x.unfold(1, factor, factor).unfold(2, factor, factor).mean(dim=[3, 4])),
    transforms.Lambda(lambda x: x.flatten(1, 2)),
    transforms.Lambda(lambda x: x.squeeze(0)),
    transforms.Lambda(lambda x: x.to(device))
])

dataset1 = tvdatasets.MNIST('./data', train=True, download=True, transform=transform)
dataset2 = tvdatasets.MNIST('./data', train=False, transform=transform)

train_loader = DataLoader(dataset1, batch_size=args.batchsize, shuffle=True)
test_loader = DataLoader(dataset2, batch_size=args.batchsize, shuffle=True)

# Take more samples from the training set? More augmentation? e.g. translate 1 pixel
train_samples, train_labels = next(iter(DataLoader(dataset1, batch_size=dataset1.data.shape[0], shuffle=True)))
test_samples, test_labels = next(iter(DataLoader(dataset2, batch_size=dataset2.data.shape[0], shuffle=False)))

train_samples = train_samples.cpu().numpy()
test_samples = test_samples.cpu().numpy()

dimension = train_samples.shape[1]

# %%
# plot settings

#TODO, swap for imported plot_samples
extra_range = 0.25 * (high_end - low_end)
cmap_low = matplotlib.cm.copper_r
cmap_high = cmr.get_sub_cmap('Greens', 0, 0.5)

# def plot_samples(samples, n_rows=4, n_cols=5):
#     assert samples.shape[0] == n_rows * n_cols, f"Expected {n_rows * n_cols} samples, got {samples.shape[0]}"

#     samples = samples.reshape(-1, side, side)

#     fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 2, n_rows * 2))
#     for i, ax in enumerate(axes.flatten() if n_rows * n_cols > 1 else [axes]):
#         if i >= samples.shape[0]:
#             break

#         image = samples[i]
#         mask_low = samples[i] > low_end
#         mask_high = samples[i] < high_end
#         masked_image_low = np.ma.masked_array(samples[i], mask=mask_low)
#         masked_image_high = np.ma.masked_array(samples[i], mask=mask_high)

#         im = ax.imshow(image, cmap="gray", vmin=low_end, vmax=high_end, interpolation='none')
#         ax.imshow(masked_image_low, cmap=cmap_low, vmin=low_end-extra_range, vmax=low_end)
#         ax.imshow(masked_image_high, cmap=cmap_high, vmin=high_end, vmax=high_end+extra_range)

#         ax.axis('off')

from utils import plot_samples as plot_samples_

def plot_samples(samples, n_rows=4, n_cols=5):
    plot_samples_(samples, n_rows, n_cols, low_end, high_end, extra_range)

# %%
# plot some samples
plot_samples(test_samples[0:20], n_rows=4, n_cols=5)
plt.savefig(save_dir / 'mnist_train_samples.png')
plt.show()
plt.close()
plt.clf()

# %%
# Basic (cross-)entropy with uniform over support
pixel_spans = (train_samples.max(0) - train_samples.min(0))
print(f"Minimum pixel span: {pixel_spans.min()}")

plt.imshow(pixel_spans.reshape(side, side), cmap='gray')
plt.colorbar()
plt.show()
plt.close()

basic_entropy = np.log(1 / 0.3081) * dimension
span_entropy = np.log(pixel_spans).sum()

print(f"Basic entropy: {basic_entropy}")
print(f"Span entropy: {span_entropy}")



# %%

#base_dist_filename = '/proj/berz-rmi/project2023/Entropy-2023/MNIST_28x28_gmm_64c_-1407.26_2023-05-02_15-14-32.pkl'
base_dist_filename = '/proj/berz-rmi/project2023/Entropy-2023/MNIST_28x28_gmm_32c_-1374.51_2023-04-27_17-22-28.pkl'
base_dist_filename = '/proj/berz-rmi/project2023/Entropy-2023/experiments/runs/MNIST_sklearn/2023-05-08_21-35-12/mixture_model.pkl'
base_dist_filename = '/proj/berz-rmi/project2023/Entropy-2023/experiments/runs/MNIST_sklearn/2023-05-08_21-35-28/mixture_model.pkl'
#base_dist_filename = '/home/x_vikni/Work/project2023/Entropy-2023/MNIST_14x14_gmm_64c_-418.60_2023-04-27_14-38-43.pkl'
#base_dist_filename = '/home/x_vikni/Work/project2023/Entropy-2023/MNIST_14x14_gmm_16c_-364.29_2023-04-27_11-02-55.pkl'
#base_dist_filename = '/home/x_vikni/Work/project2023/Entropy-2023/MNIST_14x14_gmm_1_2023-04-27_08-57-09.pkl'

#base_dist_filename = '/home/x_vikni/Work/project2023/Entropy-2023/MNIST_14x14_gmm_8_2023-04-24_16-13-26.pkl'
#base_dist_filename = '/home/x_vikni/Work/project2023/Entropy-2023/experiments/gmm/MNIST_14x14_gmm_32_2023-04-26_10-50-39.pkl'
with open(base_dist_filename, 'rb') as f:
    gmm = pickle.load(f)
train_base_dist = False
hidden_dim = [1000, 1000]
one_hot_y = True
output_type = "eT"
eps = 1e-24

use_f_div = False
use_weight_averaged_samples = False

n_epochs = 100
batchsize = 1000
sample_size = 100
lr = 0.001
lr_base_dist = 10*lr
clip_grad_norm=50.0

base_dist = KNIFE.from_gmm(gmm, device)

# Replace sample method to clip samples to support
import types

clip_samples = True

if clip_samples:
    base_dist.sample_old = base_dist.sample
    base_dist.sample = types.MethodType(
        lambda self, n: self.sample_old(n).clip(low_end, high_end)
    , base_dist)

def smooth_clip(x, low, high, eps=1e-6):
    return torch.log(torch.exp(x - low) + torch.exp(high - x) + eps) + low

#mini = MINI(base_dist, train_base_dist, hidden_dim, one_hot_y).to(device)
mini = MINI_CNN_MNIST(base_dist, train_base_dist, output_type, side, eps).to(device)

# %%

if type(mini) == REMEDI:
    optimizer = torch.optim.Adam([
        {'params': list(mini.ln.parameters())},
        {'params': mini.gc0.parameters()},
        {'params': mini.gc1.parameters()}
    ], maximize=False, lr=lr, weight_decay=0.01)
    if train_base_dist:
        optimizer.add_param_group({'params': mini.base_dist.parameters(), 'lr': lr_base_dist, 'weight_decay': 0})
else:
    optimizer = torch.optim.Adam(mini.parameters(), maximize=False, lr=lr, weight_decay=0.00001)

# %%

cross_entropies = []
for batch in test_loader:
    cross_entropies.append(-mini.base_dist.log_prob(batch[0].to(device)).detach().cpu().numpy())
    
cross_entropies = np.concatenate(cross_entropies)
cross_entropies_mean = cross_entropies.mean()
cross_entropies_std = cross_entropies.std()
print(f"Cross entropy (mean): {cross_entropies_mean:.4f} - Std: {cross_entropies_std:.4f}")

# %%

# Train model
train_losses, val_losses = train_model(mini, use_f_div, use_weight_averaged_samples, optimizer, train_loader, test_loader, n_epochs, batchsize, sample_size, clip_grad_norm, verbose=True)

# Plot losses
# Make line at cross entropy
plt.clf()
plt.axhline(cross_entropies_mean, color='r', linestyle='--', label="Cross entropy")
plt.plot(train_losses, label="Train")
plt.plot(val_losses, label="Validation")
plt.legend()
# Save plot to file
save_name = f"{save_dir}/T_MNIST_{side}x{side}_mini_{val_losses[-1]:.2f}_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.png"
plt.savefig(save_name)
plt.show()

# %%

# Save model
save_name = f"{save_dir}/T_MNIST_{side}x{side}_mini_{val_losses[-1]:.2f}_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.pt"
torch.save(mini.state_dict(), save_name)

# %%

# Load model
#save_name = "./experiments/runs/MNIST_entropy/2023-04-28_11-58-57/T_MNIST_28x28_mini_-1424.47_2023-04-28_13-18-01.pt"
save_name = "./runs/MNIST_entropy/2023-05-04_20-22-44/T_MNIST_28x28_mini_-1454.15_2023-05-04_22-29-47.pt"
mini.load_state_dict(torch.load(save_name))

# %%
# compute for one batch
batch = torch.tensor(test_samples[0:1*args.batchsize]).to(device).float()

cross_entropies = -mini.base_dist.log_prob(torch.tensor(batch))
cross_entropy = cross_entropies.mean().item()
print(f"Cross entropy (first batch): {cross_entropy:.4f}")# - Entropy upper bound: {entropy_lower_bound:.4f}")

T_batch = mini(batch).detach().cpu().numpy()
T_mean_batch = T_batch.mean()
T_std_batch = T_batch.std()
print(f"Mean T: {T_mean_batch:.4f} - Std T: {T_std_batch:.4f}")

# %%
# compute for whole dataset
cross_entropies = []
T_p = []
T_q = []#sampled on the base distribution
for batch in test_loader:
    cross_entropies.append(-mini.base_dist.log_prob(batch[0].to(device)).detach().cpu().numpy())
    T_p.append(mini(batch[0]).detach().cpu().numpy())
    batch_q = mini.base_dist.sample(sample_size)#.to(device)
    T_q.append(mini(batch_q).detach().cpu().numpy())
    
cross_entropies = np.concatenate(cross_entropies)
T_p = np.concatenate(T_p)
T_q = np.concatenate(T_q)

cross_entropies_mean = cross_entropies.mean()
cross_entropies_std = cross_entropies.std()
T_p_mean = T_p.mean()
T_p_std = T_p.std()
eT_minus_1_mean = (np.exp(T_q - 1)).mean()
eT_minus_1_std = (np.exp(T_q - 1)).std()
log_eT_mean = np.log(np.exp(T_q).mean())
print(f"Cross entropy (mean): {cross_entropies_mean:.4f} - Std: {cross_entropies_std:.4f}")
print(f"Mean T: {T_p_mean:.4f} - Std T: {T_p_std:.4f}")
print(f"Mean e^(T - 1): {eT_minus_1_mean:.4f} - Std e^(T - 1): {eT_minus_1_std:.4f}")
print(f"log(mean e^T): {log_eT_mean:.4f}")

print(f"Entropy upper bound f-div: {cross_entropies_mean - (T_p_mean - eT_minus_1_mean):.4f}")
print(f"Entropy upper bound DV: {cross_entropies_mean - (T_p_mean - log_eT_mean):.4f}")

# %%
# modified for elu
# compute for whole dataset
cross_entropies = []
T_p = []
T_q = []#sampled on the base distribution
for batch in test_loader:
    cross_entropies.append(-mini.base_dist.log_prob(batch[0].to(device)).detach().cpu().numpy())
    T_p.append(mini(batch[0]).detach().cpu().numpy())
    for _ in range(10):
        batch_q = mini.base_dist.sample(batchsize)#.to(device)
        T_q.append(mini(batch_q).detach().cpu().numpy())
    
cross_entropies = np.concatenate(cross_entropies)
T_p = np.concatenate(T_p)
T_q = np.concatenate(T_q)
comp1 = torch.mean(torch.log(torch.tensor(T_p)))
#comp2 = (torch.e ** -1) * torch.mean(torch.tensor(T_q))
comp2 = torch.log(torch.mean(torch.tensor(T_q)))

cross_entropies_mean = cross_entropies.mean()
cross_entropies_std = cross_entropies.std()
T_p_mean = T_p.mean()
T_p_std = T_p.std()
#eT_minus_1_mean = (np.exp(T_q - 1)).mean()
#eT_minus_1_std = (np.exp(T_q - 1)).std()
#log_eT_mean = np.log(np.exp(T_q).mean())
print(f"Cross entropy (mean): {cross_entropies_mean:.4f} - Std: {cross_entropies_std:.4f}")
print(f"comp1: {comp1:.4f} - comp2: {comp2:.4f}, diff: {comp1 - comp2=:.4f}")


# %%
# compute averate e^T for whole dataset

eT = []
for batch in test_loader:
    eT.append(np.exp(mini(batch[0].to(device)).detach().cpu().numpy()))

eT = np.concatenate(eT)
eT_mean = eT.mean()
eT_std = eT.std()
print(f"Mean e^T: {eT_mean:.4f} - Std e^T: {eT_std:.4f}")
# %%

n_raw = 5

raw_sample = mini.base_dist.sample(n_raw)
#raw_sample = samples[top_ind][-n_raw:]

loss_pre = mini.base_dist.log_prob(raw_sample) + mini(raw_sample)

print(f"Loss: {loss_pre.mean().item()}")
plot_samples(raw_sample.cpu(), 1, n_raw)

# %%

x = raw_sample.clone().detach().requires_grad_(True)
# optimize x wrt mini.base_dist.log_prob(x) + mini(x)

beta = 1
noise_scale = np.sqrt(2 * beta ** -1)

dt = 0.000010

saves = [x]

for i in range(1000):
    x.requires_grad = True
    #loss = mini.base_dist.log_prob(x) + torch.log(torch.nn.functional.elu(mini(x), 1) + 1 + 1e-8)
    loss = mini.base_dist.log_prob(x) + torch.log(mini(x))
    #[loss[i].backward() for i in range(n_raw)]
    loss.sum().backward()
    #x = x + 0.00001 * x.grad
    #x = x + 0.00001 * x.grad + 0.0033 * torch.randn_like(x)
    x = x + x.grad * dt + noise_scale * np.sqrt(dt) * torch.randn_like(x)
    #x = x.clip(low_end, high_end)
    #x = x + 0.001 * x.grad.sign() + 0.003 * torch.randn_like(x)
    x = x.detach()

    if i % 250 == 0:
        saves.append(x)

saves = torch.stack(saves)
plot_samples(saves.detach().cpu().flatten(0, 1), saves.shape[0], saves.shape[1])

print(f"Loss pre: {loss_pre.mean().item()}")
print(f"Loss: {loss.mean().item()}")

# put raw_sample and x in a np array and plot them

arr = np.concatenate([raw_sample.detach().cpu().numpy(), x.detach().cpu().numpy()], axis=0)
plot_samples(torch.tensor(arr).clip(low_end, high_end), 2, n_raw)

# %%

# sample n_samples and plot top n_top with highest T

n_samples = 1000
multiplier = 100
n_top = 10
#samples = mini.base_dist.sample(n_samples).clip(low_end, high_end)
sample_list = [mini.base_dist.sample(n_samples) for _ in range(multiplier)]
samples = torch.concatenate(sample_list)
T_samples_list = [mini(s).log().detach().cpu().numpy().squeeze() for s in sample_list]#squeeze dim 1 on CNN
T_samples = np.concatenate(T_samples_list)
#logprob_samples = mini.base_dist.log_prob(samples).detach().cpu().numpy()

top_ind = np.argsort(T_samples)[-n_top:]
bottom_ind = np.argsort(T_samples)[:n_top]
print(f"Top n T: {T_samples[top_ind]}")
plot_samples(samples[top_ind].cpu(), 2, 5)
plt.savefig("top_T.png")
print(f"Bottom n T: {T_samples[bottom_ind]}")
plot_samples(samples[bottom_ind].cpu(), 2, 5)
plt.savefig("bottom_T.png")
plt.show()

# %%

#threshold = 6
threshold = 6.681 # MC estimate of E_Q[T]

above_threshold = np.where(T_samples > threshold)[0]

plot_samples(samples[above_threshold].cpu().clip(low_end, high_end), 1, above_threshold.shape[0])

# %%
# sample n_samples and plot top n_top with highest logprob + T

n_samples = 3000
n_top = 50
samples = mini.base_dist.sample(n_samples)
T_samples = mini(samples).log().detach().cpu().numpy().squeeze()#squeeze dim 1 on CNN
logprob_samples = mini.base_dist.log_prob(samples).detach().cpu().numpy()

score = logprob_samples + T_samples
top_ind = np.argsort(score)[-n_top:]
bottom_ind = np.argsort(score)[:n_top]
print(f"Top n logprob + T: {logprob_samples[top_ind] + T_samples[top_ind]}")
plot_samples(samples[top_ind].cpu(), 10, 5)
plt.savefig("top_logprob_T.png")
print(f"Bottom n logprob + T: {logprob_samples[bottom_ind] + T_samples[bottom_ind]}")
plot_samples(samples[bottom_ind].cpu(), 10, 5)
plt.savefig("bottom_logprob_T.png")

# %%

beta = 5

# use softplus
def smooth_clip(x, low, high, beta=beta):
    return torch.nn.functional.softplus(x - low, beta, 20) - torch.nn.functional.softplus(x - high, beta, 20) + low

# inverse of smooth_clip
def smooth_clip_inv(y, low, high, beta=beta):
    y1 = (beta - 1) * low + torch.log(torch.exp(y) - torch.exp(torch.tensor(low)))
    y2 = torch.log(1 - torch.exp((beta - 1) * low - beta * high + y))
    return (y1 - y2) / beta

def smooth_clip_inv3(y, low, high, beta=beta):
    y1 = torch.log(torch.exp(beta * (y - low)) - 1)
    y2 = torch.log(1 - torch.exp(beta * (-high + y)))
    return low + (y1 - y2) / beta

def smooth_clip_inv4(y, low, high, beta=beta):
    y1 = torch.log(torch.exp(beta * y) - torch.exp(torch.tensor(beta * low)))
    y2 = torch.log(torch.exp(torch.tensor(beta * high)) - torch.exp(beta * y))
    return high + (y1 - y2) / beta

# derivative of smooth_clip
def smooth_clip_grad(x, low, high, beta=beta):
    return torch.sigmoid(beta * (x - low)) - torch.sigmoid(beta * (x - high))

#x = torch.linspace(-1.1, -0.9, 1000)

x = torch.linspace(-25.2, 25.2, 10000, dtype=torch.float64)
y = smooth_clip(x, -10.02, 10.02)
y_p = smooth_clip_grad(x, -10.02, 10.02)
x_inv = smooth_clip_inv4(y, -10.02, 10.02)

dx = x[1] - x[0]
y_deriv = (y[1:] - y[:-1]) / dx

plt.plot(x, y, label="smooth_clip")
plt.plot(x, x, label="identity")
plt.plot(x, x_inv, label="inverse", linestyle="dotted")
#plt.plot(x, x_inv2, label="inverse2")
plt.plot(x[:-1], y_deriv, label="num derivative")
plt.plot(x, y_p, label="smooth_clip_grad", linestyle="dotted")
plt.vlines(-1, -1.1, -0.9)
plt.legend()
plt.show()

# %%

import torch
import numpy as np
import matplotlib.pyplot as plt

beta = 15
low = -10
high = 10

# inverse of smooth_clip
def smooth_clip_inv1(y, low, high, beta=beta):
    y1 = (beta - 1) * low + torch.log(torch.exp(y) - torch.exp(torch.tensor(low)))
    y2 = torch.log(1 - torch.exp((beta - 1) * low - beta * high + y))
    #2 = torch.log(1 - torch.exp(beta * (-high + y)))
    return (y1 - y2) / beta

#def smooth_clip_inv2(y, low, high, beta=beta):
#    #1/7 (-log(e^(a + 70) - e^x) + log(e^x - e^a) + 161)
#
#    return (1 / beta) * (-torch.log(torch.exp(torch.tensor(low) + beta * (high - low)) - torch.exp(y)) + torch.log(torch.exp(y) - torch.exp(torch.tensor(low))) + beta * high)

def smooth_clip_inv3(y, low, high, beta=beta):
    y1 = torch.log(torch.exp(beta * y) - 1)
    y2 = torch.log(1 - torch.exp(beta * (low - high + y)))
    return low + (y1 - y2) / beta


def smooth_clip_inv_sep(y, low, high, beta=beta):
    mid = (low + high) / 2

    if True or y < mid:
        y = y - low
        y = beta * y
        y = torch.exp(y) - 1
        y = torch.log(y) / beta
        y = y + low
    else:
        y = y - low
        y = beta * y
        y = torch.exp(y) - 1
        y = torch.log(y) / beta
        y = y + low

    return y

# TODO: Construct inverse for "if y >= mid"



x = torch.linspace(low - 15, high + 5, 10000)
y_0 = smooth_clip_inv_sep(x, low, high)
y = smooth_clip(x, low, high, beta)
x_inv = smooth_clip_inv_sep(y, low, high)
plt.plot(x, x, label="identity")
plt.plot(x, y_0, label="smooth_clip_inv_sep")
plt.plot(x, y, label="smooth_clip")
plt.plot(x, x_inv, label="inverse", linestyle="dotted")
plt.legend()
plt.show()

# %%

mean_normalization = 0.1307 + 0.5 / 256
std_normalization = 0.3081
low_end = (0 - mean_normalization) / std_normalization
high_end = (1 - mean_normalization) / std_normalization

# standard normal density of x
std_pdf = torch.exp(-0.5 * x ** 2) / np.sqrt(2 * np.pi)

def std_pdf(x, mu=0, sigma=1):
    return torch.exp(-0.5 * ((x - mu) / sigma) ** 2) / (sigma * np.sqrt(2 * np.pi))

mean = mean_normalization
std = std_normalization
low = low_end
high = high_end
beta = 10

y_sample = np.random.normal(mean, std, 10000000)
y_sample = smooth_clip(torch.tensor(y_sample), low, high, beta).numpy()

x = torch.linspace(low-1, high+1, 10000)

y1 = std_pdf(x, mean, std)
y2 = std_pdf(smooth_clip_inv4(x, low, high, beta), mean, std) / torch.abs(smooth_clip_grad(smooth_clip_inv4(x, low, high, beta), low, high, beta))

plt.plot(x, y1, label="standard normal", linestyle="--", alpha=0.8)
plt.plot(x, y2, label="smooth clipped", linestyle="--", alpha=0.8)
#put histogram on top of plot
plt.hist(y_sample, bins=1000, density=True, label="histogram", alpha=1)
plt.xlim(-1, -0.3)
plt.legend()
plt.show()

# %%

low = 0
start = -0.01
beta = 15

x = torch.linspace(-1, 1, 10001)
y = start + torch.nn.functional.softplus(x - start, beta)

plt.plot(x, x)
plt.plot(x, y)
plt.vlines(start, -1, 1)
plt.vlines(low, -1, 1)
plt.xlim(-0.3, 0.3)
plt.show()
# %%
