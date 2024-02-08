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

import argparse

import logging

from sklearn.mixture import GaussianMixture

# %%

# load settings from command line

parser = argparse.ArgumentParser(description='MNIST sklearn')
parser.add_argument('--factor', type=int, default=1, help='Factor to downsample MNIST images by')
parser.add_argument('--n_components', type=int, default=32, help='Number of components in GMM')
parser.add_argument('--noise_scale', type=float, default=1.0, help='Standard deviation of Gaussian noise added to MNIST images')#0.3081 for old behavior
parser.add_argument('--n_repeats_train_set', type=int, default=10, help='Number of times to repeat the training set')
parser.add_argument('--max_iter', type=int, default=30, help='Maximum number of iterations for GMM')
parser.add_argument('--seed', type=int, default=7, help='Random seed')

# detect if running in jupyter notebook, and if so, use default args
if 'ipykernel' in sys.modules:
    args = parser.parse_args(args=[])
else:
    args = parser.parse_args()

print(f"{args=}")

# %%

# specify any temporary settings (for running in notebook)
#n_components = 16

# %%

save_dir = Path.cwd() / 'runs' / 'MNIST_sklearn' / datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
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

# %%

# seed RNGs

seed = args.seed
np.random.seed(seed)
torch.manual_seed(seed)

# %%

# Get MNIST dataloader

factor = args.factor
side = 28 // factor
assert side * factor == 28

noise_scale = args.noise_scale

mean_normalization = 0.1307
std_normalization = 0.3081
low_end = (0 - mean_normalization) / std_normalization
high_end = (1 - mean_normalization) / std_normalization

transform_train=transforms.Compose([
    transforms.ToTensor(),
    #Not used yet, could be added but would change the entropy too much, must then be added in test as well
    #transforms.RandomAffine(degrees=5, translate=(3/side, 1/side), interpolation=transforms.InterpolationMode.BILINEAR),#put first to avoid quantization errors
    #dequantize
    transforms.Lambda(lambda x: x + (torch.rand_like(x) - 0.5) / 255),
    #add small gaussian noise
    transforms.Lambda(lambda x: x + noise_scale * torch.randn_like(x) / 255),
    #Normalize
    transforms.Normalize((mean_normalization,), (std_normalization,)),
    #small random translation, using roll
    transforms.Lambda(lambda x: torch.roll(x, shifts=(torch.randint(-2, 3, (1,)).item(), torch.randint(-1, 2, (1,)).item()), dims=(1, 2))),
    #convert to e.g. 14x14 by taking mean of 2x2 patches
    transforms.Lambda(lambda x: x.unfold(1, factor, factor).unfold(2, factor, factor).mean(dim=[3, 4])),
    transforms.Lambda(lambda x: x.flatten(1, 2))
])

transform_test=transforms.Compose([
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x + (torch.rand_like(x) - 0.5) / 255),
    transforms.Lambda(lambda x: x + noise_scale * torch.randn_like(x) / 255),
    transforms.Normalize((mean_normalization,), (std_normalization,)),
    transforms.Lambda(lambda x: x.unfold(1, factor, factor).unfold(2, factor, factor).mean(dim=[3, 4])),
    transforms.Lambda(lambda x: x.flatten(1, 2))
])

dataset1 = tvdatasets.MNIST('./data', train=True, download=True, transform=transform_train)
dataset2 = tvdatasets.MNIST('./data', train=False, transform=transform_test)

n_repeats = args.n_repeats_train_set
train_samples, train_labels = [torch.concatenate(c) for c in list(zip(*[next(iter(torch.utils.data.DataLoader(dataset1, batch_size=dataset1.data.shape[0], shuffle=False))) for _ in range(n_repeats)]))]
test_samples, test_labels = next(iter(torch.utils.data.DataLoader(dataset2, batch_size=dataset2.data.shape[0], shuffle=False)))

train_samples = train_samples.squeeze(1).numpy()
test_samples = test_samples.squeeze(1).numpy()

dimension = train_samples.shape[1]

# %%

from utils import plot_samples as plot_samples_

extra_range = 0.25 * (high_end - low_end)
cmap_low = matplotlib.cm.copper_r
cmap_high = cmr.get_sub_cmap('Greens', 0, 0.5)

def plot_samples(samples, n_rows=4, n_cols=5):
    plot_samples_(samples, n_rows, n_cols, low_end, high_end, extra_range)

# %%
# plot some samples
plot_samples(train_samples[0:50], n_rows=10, n_cols=5)
plt.savefig(save_dir / 'mnist_train_samples.png')
plt.show()
plt.close()
plt.clf()
plot_samples(test_samples[0:50], n_rows=10, n_cols=5)
plt.savefig(save_dir / 'mnist_test_samples.png')
plt.show()
plt.close()
plt.clf()

# %%
# Basic (cross-)entropy with uniform over support
pixel_spans = (train_samples.max(0) - train_samples.min(0))
print(f"Minimum pixel span: {pixel_spans.min()}")

plt.imshow(pixel_spans.reshape(side, side), cmap='gray')
plt.colorbar()
plt.savefig(save_dir / 'mnist_train_pixel_spans.png')
plt.show()
plt.close()
plt.clf()

basic_entropy = np.log(1 / 0.3081) * dimension
span_entropy = np.log(pixel_spans).sum()

print(f"Basic entropy: {basic_entropy}")
print(f"Span entropy: {span_entropy}")



# %%
# fit gmm

n_components = args.n_components
max_iter = args.max_iter
gmm_ = GaussianMixture(n_components=n_components, covariance_type='full', reg_covar=1e-6, tol=1e-3, max_iter=max_iter, warm_start=True, verbose=2, verbose_interval=1)

# %%

gmm_.fit(train_samples)
gmm = gmm_

# %%
# Save gmm

now = datetime.now()
filename = save_dir / f"MNIST_{side}x{side}_gmm_{n_components}c_unknown_{now:%Y-%m-%d_%H-%M-%S}.pkl"
with open(filename, 'wb') as f:
    pickle.dump(gmm, f)

# %%
# get scores

train_scores = gmm.score_samples(train_samples)
test_scores = gmm.score_samples(test_samples)

# print average score on train and test set
print(f"train score: {train_scores.mean()}")
print(f"test score: {test_scores.mean()}")

n_outliers_train = (train_scores < -span_entropy).sum()
n_outliers_test = (test_scores < -span_entropy).sum()

print(f"train outliers: {n_outliers_train}")
print(f"test outliers: {n_outliers_test}")

# %%

# get posterior probabilities
test_posteriors = gmm.predict_proba(test_samples)
test_posteriors_max = np.log(test_posteriors.max(1))
test_posteriors_max.sort()
print(test_posteriors_max[:100])

# %%

samples = gmm.sample(10000)[0]
sample_posteriors = gmm.predict_proba(samples)
sample_posteriors_max = np.log(sample_posteriors.max(1))
sample_posteriors_max.sort()
print(sample_posteriors_max[:100])

# %%

cutoff = 0
bins = 100

plt.hist(train_scores.clip(cutoff, np.inf), bins=100, density=True, alpha=0.8, label="Train scores")
plt.hist(test_scores.clip(cutoff, np.inf), bins=100, density=True, alpha=0.8, label="Test scores")

labels = np.unique(train_labels)

label_avgs_train = [np.mean(train_scores[train_labels == label]) for label in labels]
label_avgs_test = [np.mean(test_scores[test_labels == label]) for label in labels]

for i, label in enumerate(labels):
    plt.annotate(label, (label_avgs_train[i], 0.0000), size=7, horizontalalignment='center', verticalalignment='bottom', bbox={"boxstyle": "circle", "color": "red", "alpha": 0.5, "pad": 0})
    plt.annotate(label, (label_avgs_test[i], 0.0002), size=7, horizontalalignment='center', verticalalignment='bottom', bbox={"boxstyle": "circle", "color": "green", "alpha": 0.5, "pad": 0})

plt.title(f"Scores {n_components} components")
plt.legend()
now = datetime.now()
plt.savefig(save_dir / f"MNIST_{side}x{side}_scores_{n_components}c_{now:%Y-%m-%d_%H-%M-%S}.png")
plt.show()
plt.close()
plt.clf()

# %%

means = gmm.means_.reshape(n_components, side, side)
covs = gmm.covariances_
vars = np.diagonal(covs, axis1=1, axis2=2).reshape(n_components, side, side)
weights = gmm.weights_

print(f"weights: {weights}")

# plot all means
rows = 2 if n_components > 1 else 1
fig, axes = plt.subplots(rows, means.shape[0] // rows, figsize=(10, 4))
for i, ax in enumerate(axes.flatten() if rows > 1 else [axes]):
    if i >= means.shape[0]:
        break

    im = ax.imshow(means[i], cmap='gray', vmin=0, vmax=1)
    ax.axis('off')
    #fig.colorbar(im)

plt.savefig(save_dir / f"means.png")
plt.show()
plt.close()
plt.clf()

# plot all pixel variances
fig, axes = plt.subplots(rows, means.shape[0] // rows, figsize=(10, 4))
for i, ax in enumerate(axes.flatten() if rows > 1 else [axes]):
    if i >= vars.shape[0]:
        break

    im = ax.imshow(vars[i])
    ax.axis('off')
    #fig.colorbar(im)

plt.savefig(save_dir / f"variances.png")
plt.show()
plt.close()
plt.clf()

# %% 
# plot samples from the gmm
n_samples = 20
samples, comps = gmm.sample(n_samples)
plot_samples(samples)
plt.savefig(save_dir / f"samples.png")
plt.show()
plt.close()
plt.clf()

# %%
# plot samples from the hottest mode of the gmm

hot_index = np.argmax(weights)
print(f"hot index: {hot_index} (weight: {weights[hot_index]}")

# plot mode and variance
fig, axes = plt.subplots(1, 2, figsize=(10, 2))
axes[0].imshow(means[hot_index], cmap='gray')
axes[0].axis('off')
axes[1].imshow(vars[hot_index])
axes[1].axis('off')

n_samples = 20
multiplier = int(3 / weights[hot_index])
samples, comps = gmm.sample(multiplier*n_samples)
samples = samples[comps == hot_index][:n_samples]

plot_samples(samples)
plt.savefig(save_dir / f"samples_hottest_mode_{weights[hot_index]}.png")
plt.show()
plt.close()
plt.clf()

# %%
# Save gmm

now = datetime.now()
filename = save_dir / f"mixture_model.pkl"
with open(filename, 'wb') as f:
    pickle.dump(gmm, f)
# %%
