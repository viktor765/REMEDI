# %%
# To be able to import from parent directory
import sys
from pathlib import Path
sys.path.append(str(Path.cwd().parent))

# %%

import numpy as np

import torch
from torch.utils.data import Dataset, DataLoader
from KNIFE import KNIFE, fit_kernel
from REMEDI import REMEDI, train_model
from synthetic import get_random_data_generator, DataGeneratorMulti
from utils import TriangleDataset
from train_complete import train_complete_model
import argparse
import copy
import pathlib
import pickle
from datetime import datetime

from matplotlib import pyplot as plt
import pandas as pd

# %%

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Provide the path to your run directory
run_dir = Path('./runs/final/2024-01-14_12-14-23/')
#assert run_dir.exists(), f"Run dir {run_dir} does not exist"

# Create folder img if it does not exist
Path('./img').mkdir(parents=True, exist_ok=True)

# %%

import io

class CPU_Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == 'torch.storage' and name == '_load_from_bytes':
            return lambda b: torch.load(io.BytesIO(b), map_location='cpu')
        else: return super().find_class(module, name)

def load_data(log_dir, device=device):
    args_data, args_REMEDI, args_KNIFE = pickle.load(open(log_dir / 'args.pkl', 'rb'))
    knife_train_losses, knife_val_losses = pickle.load(open(log_dir / 'knife_train_val_losses.pkl', 'rb'))
    remedi_train_losses, remedi_val_losses, cross_entropy = pickle.load(open(log_dir / 'remedi_train_val_losses.pkl', 'rb'))

    with open(log_dir / 'train-test_dataset.pkl', 'rb') as f:
        if device.type == 'cuda':
            train_dataset, test_dataset = pickle.load(f)
        else:
            train_dataset, test_dataset = CPU_Unpickler(f).load()

    if (log_dir / 'true_entropy.pkl').exists():
        true_entropy = pickle.load(open(log_dir / 'true_entropy.pkl', 'rb'))
    else:
        true_entropy = None

    base_dist0 = KNIFE.loadKNIFE(log_dir / 'base_dist0.pt', device=device)
    mini = torch.load(log_dir / 'remedi.pt', map_location=device)

    return {'args_data': args_data, 
            'args_REMEDI': args_REMEDI,
            'args_KNIFE': args_KNIFE,
            'KNIFE_train_losses': knife_train_losses,
            'KNIFE_val_losses': knife_val_losses,
            'REMEDI_train_losses': remedi_train_losses,
            'REMEDI_val_losses': remedi_val_losses,
            'cross_entropy': cross_entropy,
            'train_dataset': train_dataset,
            'test_dataset': test_dataset,
            'true_entropy': true_entropy,
            'base_dist0': base_dist0,
            'mini': mini}

# %%

def plot_losses(log_dir, device=device):
    data = load_data(log_dir, device=device)

    cross_entropy = data['cross_entropy']
    knife_train_losses = data['KNIFE_train_losses']
    knife_val_losses = data['KNIFE_val_losses']
    remedi_train_losses = data['REMEDI_train_losses']
    remedi_val_losses = data['REMEDI_val_losses']
    true_entropy = data['true_entropy']    

    plt.clf()
    fig, ax = plt.subplots()
    #ax.axhline(cross_entropy, color='r', linestyle='--', label="KNIFE")
    line1 = ax.plot(np.arange(1, len(knife_train_losses)+1), knife_train_losses, label="Train KNIFE")
    line2 = ax.plot(np.arange(1, len(knife_train_losses)+1), knife_val_losses, label="Validation KNIFE")
    ax.plot(np.arange(len(knife_train_losses), len(knife_train_losses)+len(remedi_train_losses)+1), [knife_train_losses[-1]] + remedi_train_losses, label="Train REMEDI", color='black', linewidth=2)
    ax.plot(np.arange(len(knife_train_losses), len(knife_train_losses)+len(remedi_train_losses)+1), [knife_val_losses[-1]] + remedi_val_losses, label="Validation REMEDI", color='red', linewidth=2)
    if true_entropy is not None:
        ax.axhline(true_entropy, color='g', linestyle='--', label="True entropy", zorder=-1)
    ax.legend()
    #plt.show()

    return fig, ax

def plot_triangle_1d(log_dir, device=device):
    data = load_data(log_dir, device=device)

    assert data['args_data'].type == 'triangle' and data['args_data'].dim == 1, "Requires 1d trinagle dataset"

    args_data = data['args_data']
    base_dist0 = data['base_dist0']
    mini = data['mini']
    train_dataset = data['train_dataset']

    x = np.linspace(-0.5, args_data.number+0.5, 10001)
    x_torch = torch.tensor(x).to(device).unsqueeze(1).float()

    q0 = np.exp(base_dist0.log_prob(x_torch).detach().cpu().numpy())
    q = np.exp(mini.base_dist.log_prob(x_torch).detach().cpu().numpy())
    T = mini(x_torch).detach().cpu().numpy()
    if mini.output_type == 'eT':
        T = np.log(T)
    qT = q*np.exp(T)

    normalizing_constant = np.trapz(qT, x)
    qTn = qT/normalizing_constant

    plt.clf()
    fig, ax = plt.subplots()
    ax.plot(*train_dataset.gen.plot(), label='p', linewidth=1)
    ax.plot(x, q0, label='q0', linewidth=1)
    #plt.plot(x, q, label='q', linewidth=1)
    ax.plot(x, qTn, label='qT (normalized)', linewidth=1)

    mu = mini.base_dist.means.squeeze(1).detach().cpu().numpy()
    weight = mini.base_dist.weigh.softmax(1).squeeze(0).detach().cpu().numpy()
    #plot weight as bars a locations mu, scaling height to fit plot
    #plt.bar(mu, 3*weight, width=0.05, alpha=0.5, label='weight')

    ax.legend()
    #plt.show()

    return fig, ax

# For testing
def analytic_KL_triangle_1d(log_dir, device=device):
    data = load_data(log_dir, device=device)

    assert data['args_data'].type == 'triangle' and data['args_data'].dim == 1, "Requires 1d trinagle dataset"

    args_data = data['args_data']
    base_dist0 = data['base_dist0']
    mini = data['mini']
    train_dataset = data['train_dataset']

    x = np.linspace(-0.5, args_data.number+0.5, 10001)
    x_torch = torch.tensor(x).to(device).unsqueeze(1).float()

    x_few, p_few = train_dataset.gen.plot()

    from scipy.interpolate import interp1d
    from scipy.special import xlogy

    # Interpolate p to x
    p_interp = interp1d(x_few, p_few, kind='linear', fill_value='extrapolate')

    p = p_interp(x)
    p[p < 0] = 0

    #q0 = np.exp(base_dist0.log_prob(x_torch).detach().cpu().numpy())
    q = np.exp(mini.base_dist.log_prob(x_torch).detach().cpu().numpy())
    T = mini(x_torch).detach().cpu().numpy()
    if mini.output_type == 'eT':
        T = np.log(T)
    qT = q*np.exp(T)

    normalizing_constant = np.trapz(qT, x)
    qTn = qT/normalizing_constant

    #plot p_interp
    plt.clf()
    fig, ax = plt.subplots()
    ax.plot(x, p, label='p_interp')
    #plt.show()

    # Compute KL
    H = -np.trapz(xlogy(p, p), x)
    CE = -np.trapz(xlogy(p, qTn), x)
    KL = CE - H
    KL2 = np.trapz(xlogy(p, p/qTn), x)

    print(f"{H=:.4f} {CE=:.4f} {KL=:.4f}")
    print(f"{KL2=:.4f}")

    return (H, CE, KL, KL2), fig, ax

# %%

def plot_two_moons_qT(log_dir, N_x:int, N_y:int, device=device):
    data = load_data(log_dir, device)
    mini = data['mini']

    x = np.linspace(-1.2, 2.2, N_x)
    y = np.linspace(-0.7, 1.2, N_y)

    xx, yy = np.meshgrid(x, y)
    xy_grid = np.stack((xx, yy))

    q = mini.base_dist.log_prob(torch.tensor(xy_grid).to(device).float().flatten(1, 2).T).reshape(xy_grid.shape[1:]).detach().cpu().numpy()
    T = mini(torch.tensor(xy_grid).to(device).float().flatten(1, 2).T).reshape(xy_grid.shape[1:]).detach().cpu().numpy()

    plt.clf()
    fig, ax = plt.subplots()
    img = ax.imshow(np.flip((T*np.exp(q)).clip(0, 3), 0), extent=[x.min(), x.max(), y.min(), y.max()])
    #plt.show()

    return fig, ax, img

def plot_two_moons_correction(log_dir, N_x:int, N_y:int, device=device):
    data = load_data(log_dir, device)
    test_dataset = data['test_dataset']
    mini = data['mini']

    x = np.linspace(-1.2, 2.2, N_x)
    y = np.linspace(-0.7, 1.2, N_y)

    xx, yy = np.meshgrid(x, y)
    xy_grid = np.stack((xx, yy))

    sample = test_dataset.samples[:1000].cpu().numpy()

    q = mini.base_dist.log_prob(torch.tensor(xy_grid).to(device).float().flatten(1, 2).T).reshape(xy_grid.shape[1:]).detach().cpu().numpy()
    T = mini(torch.tensor(xy_grid).to(device).float().flatten(1, 2).T).reshape(xy_grid.shape[1:]).detach().cpu().numpy()

    sample_T = mini(mini.base_dist.sample(1000)).detach()
    log_mean_eT = sample_T.exp().mean().log().cpu().numpy()
    print(log_mean_eT)

    plt.clf()
    fig, ax = plt.subplots()
    img = ax.imshow(np.flip(T - log_mean_eT, 0).clip(-1, 1), extent=[x.min(), x.max(), y.min(), y.max()], cmap='BrBG')
    fig.colorbar(img, orientation='horizontal', fraction=0.05, pad=0.1)
    ax.contour(np.exp(q), colors='black', levels=2, extent=[x.min(), x.max(), y.min(), y.max()])
    ax.scatter(sample[:, 0], sample[:, 1], c='red', label='samples', s=1)
    #plt.show()

    return fig, ax, img

# %%

# Generate main thesis plots
if __name__ == '__main__':
    log_parent_dir = run_dir / 'triangle_1d'
    log_dir = next(log_parent_dir.glob('*seed3'))
    assert log_dir.exists(), f"Log dir {log_dir} does not exist"
    assert Path('img').exists(), f"Dir 'img' does not exist"

    fig1, ax1 = plot_losses(log_dir, device=device)
    ax1.vlines(50, 0, 3.0, color='black', linestyle='dotted')
    ax1.set_ylim(0.75, 1.1)
    #set size of figure
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Differential entropy')
    fig1.set_size_inches(4, 3)
    ax1.legend(loc='upper right', bbox_to_anchor=(1.0, 1.0), fontsize='small')
    plt.savefig('img/loss_triangle1d.pdf', bbox_inches='tight')
    plt.savefig('img/loss_triangle1d.png', bbox_inches='tight')
    plt.show()
    fig2, ax2 = plot_triangle_1d(log_dir)
    fig2.set_size_inches(4, 3)
    ax2.legend(['$p(x)$', '$q(x)$ (KNIFE)', '$\\tilde{p}(x)$ (REMEDI)'])
    plt.savefig('img/pdf_triangle1d.pdf', bbox_inches='tight')
    plt.savefig('img/pdf_triangle1d.png', bbox_inches='tight')
    plt.show()
    # (H, CE, KL, KL2), fig3, ax3 = analytic_KL_triangle_1d(log_dir)
    # ax3.set_ylim(0, 0.5)
    # ax3.set_title(f"KL={KL:.4f}")
    # plt.show()

    # %%

    log_parent_dir = run_dir / 'triangle_8d'
    log_dir = next(log_parent_dir.glob('*seed5'))
    assert log_dir.exists(), f"Log dir {log_dir} does not exist"

    fig1, ax1 = plot_losses(log_dir, device=device)
    ax1.vlines(50, 0, 10.0, color='black', linestyle='dotted')
    ax1.set_ylim(2.0, 8.0)
    #set size of figure
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Differential entropy')
    fig1.set_size_inches(4, 3)
    ax1.legend(loc='upper right', bbox_to_anchor=(1.0, 1.0), fontsize='small')
    plt.savefig('img/loss_triangle8d.pdf', bbox_inches='tight')
    plt.savefig('img/loss_triangle8d.png', bbox_inches='tight')
    plt.show()

    # %%

    log_parent_dir = run_dir / 'triangle_20d'
    log_dir = next(log_parent_dir.glob('*seed2'))
    assert log_dir.exists(), f"Log dir {log_dir} does not exist"

    fig1, ax1 = plot_losses(log_dir, device=device)
    ax1.vlines(50, 5, 20.0, color='black', linestyle='dotted')
    ax1.set_ylim(6.0, 19.5)
    #set size of figure
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Differential entropy')
    fig1.set_size_inches(4, 3)
    ax1.legend(loc='upper right', bbox_to_anchor=(1.0, 1.0), fontsize='small')
    plt.savefig('img/loss_triangle20d.pdf', bbox_inches='tight')
    plt.savefig('img/loss_triangle20d.png', bbox_inches='tight')
    plt.show()

    # %%

    log_parent_dir = run_dir / 'two-moons_2d'
    log_dir = next(log_parent_dir.glob('*seed0'))
    assert log_dir.exists(), f"Log dir {log_dir} does not exist"

    fig1, ax1 = plot_losses(log_dir, device=device)
    ax1.vlines(50, 0, 3.0, color='black', linestyle='dotted')
    ax1.set_ylim(0.2, 1.1)
    #set size of figure
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Differential entropy')
    fig1.set_size_inches(4, 3)
    ax1.legend(loc='upper right', bbox_to_anchor=(1.0, 1.0), fontsize='small')
    plt.savefig('img/loss_two-moons.pdf', bbox_inches='tight')
    plt.savefig('img/loss_two-moons.png', bbox_inches='tight')
    plt.show()

    fig2, ax2, img2 = plot_two_moons_qT(log_dir, N_x=401, N_y=401, device=device)
    fig2.colorbar(img2, orientation='horizontal', fraction=0.05, pad=0.1)
    fig2.set_size_inches(4, 3)
    plt.savefig('img/pdf_two-moons_qT.pdf', bbox_inches='tight')
    plt.savefig('img/pdf_two-moons_qT.png', bbox_inches='tight')
    plt.show()

    fig3, ax3, img3 = plot_two_moons_correction(log_dir, N_x=401, N_y=401, device=device)
    fig3.set_size_inches(4, 3)
    plt.savefig('img/pdf_two-moons_correction.pdf', bbox_inches='tight')
    plt.savefig('img/pdf_two-moons_correction.png', bbox_inches='tight')
    plt.show()

    # %%

    log_parent_dir = run_dir / 'ball_8d'
    log_dir = next(log_parent_dir.glob('*seed3'))
    assert log_dir.exists(), f"Log dir {log_dir} does not exist"
    assert Path('img').exists(), f"Dir 'img' does not exist"

    fig1, ax1 = plot_losses(log_dir, device=device)
    ax1.vlines(10, -10.0, 10.0, color='black', linestyle='dotted')
    ax1.set_ylim(-0.2, 1.5)
    #set size of figure
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Differential entropy')
    fig1.set_size_inches(4, 3)
    ax1.legend(loc='upper right', bbox_to_anchor=(1.0, 1.0), fontsize='small')
    plt.savefig('img/loss_ball8d.pdf', bbox_inches='tight')
    plt.savefig('img/loss_ball8d.png', bbox_inches='tight')
    plt.show()

    # %%

    log_parent_dir = run_dir / 'ball_20d'
    log_dir = next(log_parent_dir.glob('*seed3'))
    assert log_dir.exists(), f"Log dir {log_dir} does not exist"
    assert Path('img').exists(), f"Dir 'img' does not exist"

    fig1, ax1 = plot_losses(log_dir, device=device)
    ax1.vlines(10, -10.0, 10.0, color='black', linestyle='dotted')
    ax1.set_ylim(-0.2, 2.0)
    #set size of figure
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Differential entropy')
    fig1.set_size_inches(4, 3)
    ax1.legend(loc='upper right', bbox_to_anchor=(1.0, 1.0), fontsize='small')
    plt.savefig('img/loss_ball20d.pdf', bbox_inches='tight')
    plt.savefig('img/loss_ball20d.png', bbox_inches='tight')
    plt.show()

    # %%

    log_parent_dir = run_dir / 'cube_8d'
    log_dir = next(log_parent_dir.glob('*seed3'))
    assert log_dir.exists(), f"Log dir {log_dir} does not exist"
    assert Path('img').exists(), f"Dir 'img' does not exist"

    fig1, ax1 = plot_losses(log_dir, device=device)
    ax1.vlines(10, -10.0, 10.0, color='black', linestyle='dotted')
    ax1.set_ylim(-0.2, 2.5)
    #set size of figure
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Differential entropy')
    fig1.set_size_inches(4, 3)
    ax1.legend(loc='upper right', bbox_to_anchor=(1.0, 1.0), fontsize='small')
    plt.savefig('img/loss_cube8d.pdf', bbox_inches='tight')
    plt.savefig('img/loss_cube8d.png', bbox_inches='tight')
    plt.show()

    # %%

    log_parent_dir = run_dir / 'cube_20d'
    log_dir = next(log_parent_dir.glob('*seed3'))
    assert log_dir.exists(), f"Log dir {log_dir} does not exist"
    assert Path('img').exists(), f"Dir 'img' does not exist"

    fig1, ax1 = plot_losses(log_dir, device=device)
    ax1.vlines(10, -10.0, 10.0, color='black', linestyle='dotted')
    ax1.set_ylim(-0.2, 5.5)
    #set size of figure
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Differential entropy')
    fig1.set_size_inches(4, 3)
    ax1.legend(loc='upper right', bbox_to_anchor=(1.0, 1.0), fontsize='small')
    plt.savefig('img/loss_cube20d.pdf', bbox_inches='tight')
    plt.savefig('img/loss_cube20d.png', bbox_inches='tight')
    plt.show()

# %%

# Make dataframe with all averages and stds for training and validation losses
if __name__ == "__main__":
    log_parent_dirs = [
        run_dir / 'triangle_8d_varcomps_network',
        run_dir / 'triangle_8d_varcomps',
        run_dir / 'ball_8d_varcomps',
        run_dir / 'cube_8d_varcomps',
        run_dir / 'triangle_20d_varcomps',
        run_dir / 'ball_20d_varcomps',
        run_dir / 'cube_20d_varcomps',
        run_dir / 'ball_8d',
        run_dir / 'ball_20d',
        run_dir / 'cube_8d',
        run_dir / 'cube_20d',
        run_dir / 'triangle_1d',
        run_dir / 'triangle_8d',
        run_dir / 'two-moons_2d'
    ]

    log_dirs = [list(log_parent_dir.glob('*seed*')) for log_parent_dir in log_parent_dirs]
    log_dirs = sum(log_dirs, [])#join all lists
    log_dirs = sorted(log_dirs, key=lambda x: int(x.name.split('seed')[1]))

    df = pd.DataFrame(columns=['type', 'true_entropy', 'seed', 'n_components', 'KNIFE_train_losses', 'KNIFE_val_losses', 'REMEDI_train_losses', 'REMEDI_val_losses'])

    for log_dir in log_dirs:
        try:    
            data = load_data(log_dir, device=device)
            df.loc[len(df)] = [
                log_dir.parent.name,
                data['true_entropy'],
                int(log_dir.name.split('seed')[1]),
                data['args_KNIFE'].num_modes,
                np.array(data['KNIFE_train_losses']),
                np.array(data['KNIFE_val_losses']),
                np.array(data['REMEDI_train_losses']),
                np.array(data['REMEDI_val_losses'])
            ]
        except:
            print(f'Error with {log_dir}')

    #sort by number of components and then by seed
    df = df.sort_values(by=['n_components', 'seed'])

    #average over seeds
    df_avg = df.groupby(['type', 'n_components']).mean()

    #add std, keeping in mind that 'KNIFE_train_losses' is a np.array
    df_avg['KNIFE_train_losses_std'] = df.groupby(['type', 'n_components'])['KNIFE_train_losses'].apply(lambda x: np.std(np.vstack(x), axis=0))
    df_avg['KNIFE_val_losses_std'] = df.groupby(['type', 'n_components'])['KNIFE_val_losses'].apply(lambda x: np.std(np.vstack(x), axis=0))
    df_avg['REMEDI_train_losses_std'] = df.groupby(['type', 'n_components'])['REMEDI_train_losses'].apply(lambda x: np.std(np.vstack(x), axis=0))
    df_avg['REMEDI_val_losses_std'] = df.groupby(['type', 'n_components'])['REMEDI_val_losses'].apply(lambda x: np.std(np.vstack(x), axis=0))

    # make dataframe with last elements of each list in df_avg, i.e. the final estimates
    df_avg_last = pd.DataFrame(columns=['type', 'true_entropy', 'n_components', 'KNIFE_train_mean', 'KNIFE_val_mean', 'REMEDI_train_mean', 'REMEDI_val_mean', 'KNIFE_train_std', 'KNIFE_val_std', 'REMEDI_train_std', 'REMEDI_val_std'])

    for i in range(len(df_avg)):
        has_remedi = len(df_avg.iloc[i]['REMEDI_train_losses']) > 0

        df_avg_last.loc[len(df_avg_last)] = [
            df_avg.index[i][0],
            df_avg.iloc[i]['true_entropy'],
            df_avg.index[i][1],
            df_avg.iloc[i]['KNIFE_train_losses'][-1],
            df_avg.iloc[i]['KNIFE_val_losses'][-1],
            df_avg.iloc[i]['REMEDI_train_losses'][-1] if has_remedi else None,
            df_avg.iloc[i]['REMEDI_val_losses'][-1] if has_remedi else None,
            df_avg.iloc[i]['KNIFE_train_losses_std'][-1],
            df_avg.iloc[i]['KNIFE_val_losses_std'][-1],
            df_avg.iloc[i]['REMEDI_train_losses_std'][-1] if has_remedi else None,
            df_avg.iloc[i]['REMEDI_val_losses_std'][-1] if has_remedi else None
        ]

    # Save to csv
    df.to_csv('img/big.csv')
    df_avg.to_csv('img/avg_big.csv')
    df_avg_last.to_csv('img/avg_last_big.csv')

    # %%

    # Load dataframe with all averages and stds for training and validation losses
    # This may be useful if you do not have the runs, but the csv files
  
    load_from_csv = False
    if load_from_csv:
        df_avg = pd.read_csv('avg_big.csv', index_col=[0, 1])
        df_avg_last = pd.read_csv('avg_last_big.csv', index_col=[0, 1])

        # Remember to parse array columns of df_avg
        for col in ['KNIFE_train_losses', 'KNIFE_val_losses', 'REMEDI_train_losses', 'REMEDI_val_losses', 'KNIFE_train_losses_std', 'KNIFE_val_losses_std', 'REMEDI_train_losses_std', 'REMEDI_val_losses_std']:
            df_avg[col] = df_avg[col].apply(lambda x: np.fromstring(x[1:-1], sep=' '))

    # %%

    # Plot averaged losses and std-error bars for variable number of components

    show_train = True

    # Choose which group to plot
    group = 'triangle_8d_varcomps_network'
    group = 'triangle_8d_varcomps'
    group = 'ball_8d_varcomps'
    group = 'cube_8d_varcomps'
    group = 'triangle_20d_varcomps'
    group = 'ball_20d_varcomps'
    group = 'cube_20d_varcomps'

    print(group)

    # plot
    fig, ax = plt.subplots(figsize=(4, 3))
    df_avg_group = df_avg.loc[group]
    for n_components in df_avg_group.index:
        knife_train_losses = df_avg_group.loc[(n_components)]['KNIFE_train_losses']
        knife_train_losses_std = df_avg_group.loc[(n_components)]['KNIFE_train_losses_std']
        knife_val_losses = df_avg_group.loc[(n_components)]['KNIFE_val_losses']
        knife_val_losses_std = df_avg_group.loc[(n_components)]['KNIFE_val_losses_std']
        remedi_train_losses = df_avg_group.loc[(n_components)]['REMEDI_train_losses']
        remedi_train_losses_std = df_avg_group.loc[(n_components)]['REMEDI_train_losses_std']
        remedi_val_losses = df_avg_group.loc[(n_components)]['REMEDI_val_losses']
        remedi_val_losses_std = df_avg_group.loc[(n_components)]['REMEDI_val_losses_std']

        line_val = ax.plot(np.arange(1, len(knife_train_losses)+1), knife_val_losses, label=f"{n_components} comp.")
        #plot std
        ax.fill_between(np.arange(1, len(knife_train_losses)+1), knife_val_losses - knife_val_losses_std, knife_val_losses + knife_val_losses_std, alpha=0.2, color=line_val[-1].get_color())
        if show_train:
            line_train = ax.plot(np.arange(1, len(knife_train_losses)+1), knife_train_losses, label="_nolegend_", alpha=0.3, color=line_val[-1].get_color(), linestyle='dotted')
            #plot std
            ax.fill_between(np.arange(1, len(knife_train_losses)+1), knife_train_losses - knife_train_losses_std, knife_train_losses + knife_train_losses_std, alpha=0.08, color=line_train[-1].get_color())

        ax.plot(np.arange(len(knife_train_losses), len(knife_train_losses)+len(remedi_train_losses)+1), np.hstack((knife_val_losses[-1], remedi_val_losses)), label="_nolegend_", color=line_val[-1].get_color())
        #plot std
        ax.fill_between(np.arange(len(knife_train_losses)+1, len(knife_train_losses)+len(remedi_train_losses)+1), remedi_val_losses - remedi_val_losses_std, remedi_val_losses + remedi_val_losses_std, alpha=0.2, color=line_val[-1].get_color())
        if show_train:
            ax.plot(np.arange(len(knife_train_losses), len(knife_train_losses)+len(remedi_train_losses)+1), np.hstack((knife_train_losses[-1], remedi_train_losses)), label="_nolegend_", alpha=0.3, color=line_val[-1].get_color(), linestyle='dotted')
            #plot std
            ax.fill_between(np.arange(len(knife_train_losses)+1, len(knife_train_losses)+len(remedi_train_losses)+1), remedi_train_losses - remedi_train_losses_std, remedi_train_losses + remedi_train_losses_std, alpha=0.08, color=line_val[-1].get_color())

    assert np.all(df_avg_group['true_entropy'] == df_avg_group['true_entropy'][1])
    true_entropy = df_avg_group['true_entropy'][1]
    if true_entropy is not None:
        ax.axhline(true_entropy, color='g', linestyle='--', label="True entropy", zorder=-1)
  
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Differential entropy')

    if group == 'triangle_8d_varcomps_network':
        ax.vlines(len(knife_train_losses), 0.0, len(knife_train_losses), color='black', linestyle='dotted')
        ax.set_ylim([2.5, 6.1])
        ax.legend(loc='upper right', bbox_to_anchor=(1.0, 1.0), ncol=1, fontsize='x-small', columnspacing=0.5, framealpha=0.9)
    elif group == 'triangle_8d_varcomps':
        ax.set_ylim([2.2, 7.4])
        ax.legend(loc='upper right', bbox_to_anchor=(1.0, 1.0), ncol=2, fontsize='x-small', columnspacing=0.5, framealpha=0.9)
    elif group == 'ball_8d_varcomps':
        ax.set_ylim([-0.3, 1.1])
        ax.legend(loc='lower left', bbox_to_anchor=(0.02, 0.02), ncol=2, fontsize='x-small', columnspacing=0.5, framealpha=0.9)
    elif group == 'cube_8d_varcomps':
        ax.set_ylim([-0.4, 2.1])
        ax.legend(loc='lower left', bbox_to_anchor=(0.03, 0), ncol=2, fontsize='x-small', columnspacing=0.5, framealpha=0.9)
    elif group == 'ball_20d_varcomps':
        ax.set_ylim([-0.3, 1.6])
        ax.legend(loc='lower left', bbox_to_anchor=(0.02, 0.02), ncol=2, fontsize='x-small', columnspacing=0.5, framealpha=0.9)
    elif group == 'cube_20d_varcomps':
        ax.set_ylim([-0.3, 4.3])
        ax.legend(loc='lower left', bbox_to_anchor=(0.03, 0), ncol=2, fontsize='x-small', columnspacing=0.5, framealpha=0.9)
    elif group == 'triangle_20d_varcomps':
        ax.set_ylim([5.8, 19.5])
        ax.legend(loc='upper right', bbox_to_anchor=(1.0, 1.0), ncol=2, fontsize='x-small', columnspacing=0.5, framealpha=0.9)

    #plt.show()

    fig.tight_layout()
    fig.savefig(f'img/{group}_losses.pdf')
    fig.savefig(f'img/{group}_losses.png')

    if group == 'triangle_8d_varcomps_network':
        # Use only legend from first figure in paper
        #ax.legend(loc='lower left', fontsize='small', ncol=1, framealpha=0.5)
        ax.legend().remove()
        ax.set_ylim([2.9, 3.7])
        ax.set_xlim([len(knife_train_losses) - 10, None])
        fig.tight_layout()
        fig.savefig(f'img/{group}_losses_zoom.pdf')
        fig.savefig(f'img/{group}_losses_zoom.png')
    
    #plt.show()

    # %%

    # Plot averaged losses and std-error bars for non-variable number of components

    show_train = True

    # Choose which group to plot
    group = 'ball_8d'
    group = 'ball_20d'
    group = 'cube_8d'
    group = 'cube_20d'
    group = 'triangle_1d'
    group = 'triangle_8d'
    group = 'two-moons_2d'

    print(group)

    # plot
    fig, ax = plt.subplots(figsize=(4, 3))
    ax.set_ylim([10.9999, 14.1])
    df_avg_group = df_avg.loc[group]
    for n_components in df_avg_group.index:
        knife_train_losses = df_avg_group.loc[(n_components)]['KNIFE_train_losses']
        knife_train_losses_std = df_avg_group.loc[(n_components)]['KNIFE_train_losses_std']
        knife_val_losses = df_avg_group.loc[(n_components)]['KNIFE_val_losses']
        knife_val_losses_std = df_avg_group.loc[(n_components)]['KNIFE_val_losses_std']
        remedi_train_losses = df_avg_group.loc[(n_components)]['REMEDI_train_losses']
        remedi_train_losses_std = df_avg_group.loc[(n_components)]['REMEDI_train_losses_std']
        remedi_val_losses = df_avg_group.loc[(n_components)]['REMEDI_val_losses']
        remedi_val_losses_std = df_avg_group.loc[(n_components)]['REMEDI_val_losses_std']

        line_val = ax.plot(np.arange(1, len(knife_train_losses)+1), knife_val_losses, label="Validation KNIFE", color='#ff7f0e')
        #plot std
        ax.fill_between(np.arange(1, len(knife_train_losses)+1), knife_val_losses - knife_val_losses_std, knife_val_losses + knife_val_losses_std, alpha=0.2, color=line_val[-1].get_color())
        if show_train:
            None

        ax.plot(np.arange(len(knife_train_losses), len(knife_train_losses)+len(remedi_train_losses)+1), np.hstack((knife_val_losses[-1], remedi_val_losses)), label="Validation REMEDI", color='black')
        #plot std
        ax.fill_between(np.arange(len(knife_train_losses)+1, len(knife_train_losses)+len(remedi_train_losses)+1), remedi_val_losses - remedi_val_losses_std, remedi_val_losses + remedi_val_losses_std, alpha=0.2, color='black')
        
        if show_train:
            line_train = ax.plot(np.arange(1, len(knife_train_losses)+1), knife_train_losses, label="Train KNIFE", alpha=0.3, color='#1f77b4', linestyle='dotted')
            #plot std
            ax.fill_between(np.arange(1, len(knife_train_losses)+1), knife_train_losses - knife_train_losses_std, knife_train_losses + knife_train_losses_std, alpha=0.08, color=line_train[-1].get_color())

            ax.plot(np.arange(len(knife_train_losses), len(knife_train_losses)+len(remedi_train_losses)+1), np.hstack((knife_train_losses[-1], remedi_train_losses)), label="Train REMEDI", alpha=0.3, color='red', linestyle='dotted')
            #plot std
            ax.fill_between(np.arange(len(knife_train_losses)+1, len(knife_train_losses)+len(remedi_train_losses)+1), remedi_train_losses - remedi_train_losses_std, remedi_train_losses + remedi_train_losses_std, alpha=0.08, color='red')

    assert np.all(df_avg_group['true_entropy'] == df_avg_group['true_entropy'].iloc[0]) or np.isnan(df_avg_group['true_entropy'].iloc[0])
    true_entropy = df_avg_group['true_entropy'].iloc[0]
    if true_entropy is not None:
        ax.axhline(true_entropy, color='g', linestyle='--', label="True entropy", zorder=-1)
  
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Differential entropy')

    if group in ['ball_8d', 'ball_20d', 'cube_8d', 'cube_20d']:
        best_knife_val = df_avg.loc[f'{group}_varcomps', 256]['KNIFE_val_losses'][:len(knife_val_losses)+len(remedi_val_losses)] 
        best_knife_val_std = df_avg.loc[f'{group}_varcomps', 256]['KNIFE_val_losses_std'][:len(knife_val_losses)+len(remedi_val_losses)]
        ax.plot(np.arange(1, len(best_knife_val)+1), best_knife_val, label="256 comp. KNIFE", alpha=0.5, color='purple', linestyle='--', zorder=-1)
        #plot std
        ax.fill_between(np.arange(1, len(best_knife_val)+1), best_knife_val - best_knife_val_std, best_knife_val + best_knife_val_std, alpha=0.1, color='brown', zorder=-1)

    if group == 'ball_8d':
        ax.set_ylim([-0.1, 1.1])
        ax.legend(loc='upper right', bbox_to_anchor=(1.0, 1.0), ncol=1, fontsize='x-small', columnspacing=0.5, framealpha=0.9)
    elif group == 'ball_20d':
        ax.set_ylim([-0.1, 2.2])
        ax.legend(loc='upper right', bbox_to_anchor=(1.0, 1.0), ncol=1, fontsize='x-small', columnspacing=0.5, framealpha=0.9)
    elif group == 'cube_8d':
        ax.set_ylim([-0.1, 2.2])
        ax.legend(loc='upper right', bbox_to_anchor=(1.0, 1.0), ncol=1, fontsize='x-small', columnspacing=0.5, framealpha=0.9)
    elif group == 'cube_20d':
        ax.set_ylim([-0.2, 5.6])
        ax.legend(loc='upper right', bbox_to_anchor=(1.0, 1.0), ncol=1, fontsize='x-small', columnspacing=0.5, framealpha=0.9)
    elif group == 'triangle_1d':
        ax.set_ylim([0.75, 1.1])
        ax.legend(loc='upper right', bbox_to_anchor=(1.0, 1.0), ncol=1, fontsize='x-small', columnspacing=0.5, framealpha=0.9)
    elif group == 'triangle_8d':
        ax.set_ylim([2.2, 7.4])
        ax.legend(loc='upper right', bbox_to_anchor=(1.0, 1.0), ncol=1, fontsize='x-small', columnspacing=0.5, framealpha=0.9)
    elif group == 'two-moons_2d':
        ax.set_ylim([0.2, 0.9])
        ax.legend(loc='upper right', bbox_to_anchor=(1.0, 1.0), ncol=1, fontsize='x-small', columnspacing=0.5, framealpha=0.9)
    
    ax.vlines(len(knife_train_losses), -1.0, len(knife_train_losses), color='black', linestyle='dotted')

    plt.show()

    fig.tight_layout()
    fig.savefig(f'img/{group}_losses.pdf')
    fig.savefig(f'img/{group}_losses.png')

# %%
