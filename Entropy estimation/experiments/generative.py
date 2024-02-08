# %%
# To be able to import from parent directory
import sys
from pathlib import Path
sys.path.append(str(Path.cwd().parent))

# %%

import numpy as np
import torch
from datetime import datetime
from matplotlib import pyplot as plt

from post_plots import load_data

# %%

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Create folder img if it does not exist
Path('./img').mkdir(parents=True, exist_ok=True)

# %%

run_dir = Path('./runs/final/2024-01-14_12-14-23/')
assert run_dir.exists(), f"Run dir {run_dir} does not exist"

log_parent_dir = run_dir / 'two-moons_2d'
log_dir = next(log_parent_dir.glob('*seed0'))

data = load_data(log_dir, device)
test_dataset = data['test_dataset']
mini = data['mini']

T_dataset = mini(test_dataset.samples[:50000]).detach()

C = T_dataset.max().item() + 0.05
print(f'{C=}')

# %%

# Rejection sampling

# seed
torch.manual_seed(7)

n_samples = 10000

samples = mini.base_dist.sample((n_samples,)).detach()
T = mini(samples).detach()
T = T / C

# draw uniform samples, accept if uniform sample is less than T
uniform_samples = torch.rand((n_samples,), device=device)
accepted = uniform_samples < T

# accepted samples
accepted_samples = samples[accepted].cpu().numpy()
rejected_samples = samples[~accepted].cpu().numpy()
print(f'{len(accepted_samples)=}')

#plot samples
fig1, ax_proposal = plt.subplots(figsize=(4, 3))
ax_proposal.scatter(samples[:, 0].cpu(), samples[:, 1].cpu(), s=2, label='proposed')
ax_proposal.legend()
plt.savefig('img/two_moons_proposed_samples.pdf', bbox_inches='tight')
plt.savefig('img/two_moons_proposed_samples.png', bbox_inches='tight')
plt.show()
fig2, ax2 = plt.subplots(figsize=(4, 3))

x_limits = ax_proposal.get_xlim()
y_limits = ax_proposal.get_ylim()

ax2.scatter(accepted_samples[:, 0], accepted_samples[:, 1], s=2, label='accepted')
ax2.legend()
# set axis limits to same as above
ax2.set_xlim(x_limits)
ax2.set_ylim(y_limits)
plt.savefig('img/two_moons_accepted_samples.pdf', bbox_inches='tight')
plt.savefig('img/two_moons_accepted_samples.png', bbox_inches='tight')
plt.show()

# %%

# Sample using Langevin dynamics

# seed
torch.manual_seed(7)

n_samples = 10000
raw_sample = mini.base_dist.sample((n_samples,)).detach()

x = raw_sample.clone().detach().requires_grad_(True)

dt = 0.001
n_steps = 100

betas = [0.1, 0.3, 1.0, 3.0, 10]
saves = []
save_every = 10

for beta in betas:
    saves.append([x])
    noise_scale = np.sqrt(2 * beta ** -1)
    for i in range(n_steps):
        x.requires_grad = True
        loss = mini.base_dist.log_prob(x) + torch.log(mini(x))
        loss.sum().backward()
        x = x + x.grad * dt + noise_scale * np.sqrt(dt) * torch.randn_like(x)
        x = x.detach()

        if i % save_every == 0 and i != 0:
            saves[-1].append(x)

# %%

fig, axs = plt.subplots(1, len(betas)+1, figsize=((len(betas)+1)*3, 3))
axs[0].scatter(raw_sample[:, 0].cpu(), raw_sample[:, 1].cpu(), s=1)
axs[0].set_xlim(x_limits)
axs[0].set_ylim(y_limits)
axs[0].set_title('$X_0$')
for ax, beta, save in zip(axs[1:], betas, saves):
    x = save[-1].detach()
    ax.scatter(x[:, 0].cpu(), x[:, 1].cpu(), s=1)
    ax.set_xlim(x_limits)
    ax.set_ylim(y_limits)
    ax.set_title(f'$\\beta={beta}$')

plt.savefig('img/two_moons_sde_samples.pdf', bbox_inches='tight')
plt.savefig('img/two_moons_sde_samples.png', bbox_inches='tight')

# %%

