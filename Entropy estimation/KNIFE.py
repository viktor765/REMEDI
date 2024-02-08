import numpy as np
import torch
import torch.nn as nn
import torch.distributions as D

from types import SimpleNamespace
from torch.utils.data import DataLoader
from tqdm import tqdm
from contextlib import nullcontext

from sklearn.mixture import GaussianMixture

class MargKernel(nn.Module):
    def __init__(self, args, zc_dim, zd_dim, init_samples=None):

        self.optimize_mu = args.optimize_mu
        self.K = args.marg_modes if self.optimize_mu else args.batch_size
        self.d = zc_dim
        self.use_tanh = args.use_tanh
        self.init_std = args.init_std
        super(MargKernel, self).__init__()

        self.logC = torch.tensor([-self.d / 2 * np.log(2 * np.pi)])

        self.dtype = torch.float32

        if init_samples is None:
            init_samples = self.init_std * torch.randn(self.K, self.d)
        init_samples = init_samples.to(self.dtype)
        # self.means = nn.Parameter(torch.rand(self.K, self.d), requires_grad=True)
        if self.optimize_mu:
            self.means = nn.Parameter(init_samples, requires_grad=True)  # [K, db]
        else:
            self.means = nn.Parameter(init_samples, requires_grad=False)

        if args.cov_diagonal == 'var':
            diag = self.init_std * torch.randn((1, self.K, self.d), dtype=self.dtype)
        else:
            diag = self.init_std * torch.randn((1, 1, self.d), dtype=self.dtype)
        self.logvar = nn.Parameter(diag, requires_grad=True)

        if args.cov_off_diagonal == 'var':
            tri = self.init_std * torch.randn((1, self.K, self.d, self.d))
            tri = tri.to(init_samples.dtype)
            self.tri = nn.Parameter(tri, requires_grad=True)
        else:
            self.tri = None

        weigh = torch.ones((1, self.K), dtype=self.dtype)
        if args.average == 'var':
            self.weigh = nn.Parameter(weigh, requires_grad=True)
        else:
            self.weigh = nn.Parameter(weigh, requires_grad=False)

    def logpdf(self, x):
        #print(x.shape)
        #assert len(x.shape) == 2 and x.shape[1] == self.d, 'x has to have shape [N, d]'
        #print(f"self.d = {self.d}, x.shape[1] = {x.shape[1]}")
        assert x.shape[1] == self.d, 'x has to have shape [N, d]'
        assert x.dtype == self.dtype, f'x has to be of type {self.dtype}'

        x = x[:, None, :]
        w = torch.log_softmax(self.weigh, dim=1)
        y = x - self.means
        logvar = self.logvar
        if self.use_tanh:
            logvar = logvar.tanh()
        var = logvar.exp()
        y = y * var
        # print(f"Marg : {var.min()} | {var.max()} | {var.mean()}")
        if self.tri is not None:
            y = y + torch.squeeze(torch.matmul(torch.tril(self.tri, diagonal=-1), y[:, :, :, None]), 3)
        y = torch.sum(y ** 2, dim=2)

        y = -y / 2 + torch.sum(torch.log(torch.abs(var) + 1e-8), dim=-1) + w
        y = torch.logsumexp(y, dim=-1)
        return self.logC.to(y.device) + y
    
    # Alias for logpdf, for compatibility with torch.distributions
    def log_prob(self, x):
        return self.logpdf(x)

    def sample(self, n):
        # For compatibility with torch.distributions
        if type(n) is tuple:
            n = n[0]

        weights = torch.softmax(self.weigh, dim=1)[0]
        mu = self.means
        var = self.logvar.exp()
        tri = torch.tril(self.tri, diagonal=-1)

        V = torch.diag_embed(var)
        L = V + tri @ V#Cholesky factor of precision matrix
        
        I = D.Categorical(weights).sample((n,))

        mu_I = mu[I]
        L_I = L.squeeze(0)[I]

        d = D.MultivariateNormal(torch.zeros_like(mu_I).to(mu_I.device), torch.eye(mu_I.shape[1]).to(mu_I.device))
        z = d.sample()
        z1 = torch.linalg.solve_triangular(L_I, z.unsqueeze(-1), upper=False).squeeze(-1) 
        samples = z1 + mu_I

        return samples

    def update_parameters(self, z):
        self.means = z

    def forward(self, x):
        y = -self.logpdf(x)
        return torch.mean(y)
    
class KNIFE(MargKernel):
    def __init__(self, args, zc_dim, zd_dim, init_samples=None):
        super(KNIFE, self).__init__(args, zc_dim, zd_dim, init_samples)

    def weight_averaged_sample_function(self, T, n):
        if type(n) is tuple:
            n = n[0]

        weights = torch.softmax(self.weigh, dim=1).squeeze(0)
        mu = self.means
        var = self.logvar.exp()
        tri = torch.tril(self.tri, diagonal=-1)

        V = torch.diag_embed(var)
        L = (V + tri @ V).squeeze(0)#Cholesky factor of precision matrix
            
        d = D.MultivariateNormal(torch.zeros_like(mu).to(mu.device), torch.eye(mu.shape[1]).to(mu.device))

        z = d.sample((n,))
        # print(z.shape)
        z1 = torch.linalg.solve_triangular(L, z.unsqueeze(-1), upper=False).squeeze(-1) 
        samples = z1 + mu
        # print(z1.shape)
        # print(T(samples).shape)
        # print(torch.linalg.solve_triangular(L, z.unsqueeze(-1), upper=False).shape)

        #collapse the first two dimensions into one
        samples_flattened = torch.flatten(samples, 0, 1)

        flattened_T_res = T(samples_flattened)
        
        #unflatten the first dimension into the first two dimensions
        T_res = flattened_T_res.view(n, self.K)

        return T_res @ weights
    
    @staticmethod
    def loadKNIFE(path: str, device = 'cpu'):
        params = torch.load(path, map_location=device)

        args = SimpleNamespace(
            optimize_mu=True,
            marg_modes=params['means'].shape[0],
            use_tanh=False,
            init_std=1e-2,
            cov_diagonal='var',
            cov_off_diagonal='var',
            average='var')

        d = params['means'].shape[1]

        model = KNIFE(args, d, 42).to(device)
        model.load_state_dict(params)

        return model        
    
    @staticmethod
    def from_params(means, logvar, tri, weigh, device = 'cpu'):
        args = SimpleNamespace(
            optimize_mu=True,
            marg_modes=means.shape[0],
            use_tanh=False,
            init_std=1e-2,
            cov_diagonal='var',
            cov_off_diagonal='var',
            average='var')

        d = means.shape[1]

        model = KNIFE(args, d, 42).to(device)

        assert means.shape == model.means.shape
        assert logvar.shape == model.logvar.shape
        assert tri.shape == model.tri.shape
        assert weigh.shape == model.weigh.shape

        model.means = torch.nn.Parameter(means.float(), requires_grad=args.optimize_mu)
        model.logvar = torch.nn.Parameter(logvar.float(), requires_grad=True)
        model.tri = torch.nn.Parameter(tri.float(), requires_grad=True)
        model.weigh = torch.nn.Parameter(weigh.float(), requires_grad=True)

        return model
    
    @staticmethod
    def from_params_chol(means, L, weights, device = 'cpu'):
        means = means.float()
        L = L.float()
        weights = weights.float()

        diag = L.diagonal(dim1=-2, dim2=-1)
        logvar = torch.log(diag)
        r_tri = ((L - torch.diag_embed(diag)) @ (torch.diag_embed(1.0 / diag))).tril(diagonal=-1)

        return KNIFE.from_params(means, logvar.unsqueeze(0), r_tri.unsqueeze(0), weights.log().unsqueeze(0), device)
    
    @staticmethod
    def from_gmm(gmm: GaussianMixture, device = 'cpu'):
        means = torch.tensor(gmm.means_).float()
        L = torch.tensor(gmm.precisions_cholesky_).transpose(1, 2).float()
        weights = torch.tensor(gmm.weights_).float()

        return KNIFE.from_params_chol(means, L, weights, device)

def fit_kernel(args, print_log, return_history = False):
    if type(args.train_samples) is torch.utils.data.dataloader.DataLoader:
        dataloader = args.train_samples
        test_loader = args.test_samples
    else:
        dataloader = DataLoader(args.train_samples, batch_size=args.batchsize, shuffle=args.shuffle, drop_last=False)
        test_loader = DataLoader(args.test_samples, batch_size=args.batchsize, shuffle=False, drop_last=False)

    #load at least args.num_modes samples using the dataloader
    tmp = []
    iterator = iter(dataloader)
    for _ in range(args.num_modes // dataloader.batch_size + 1):
        tmp.append(next(iterator))

    kernel_samples = torch.cat(tmp, dim=0)[:args.num_modes]
    #kernel_samples = args.train_samples[np.random.choice(len(args.train_samples), args.num_modes, replace=False)]

    args_adaptive = SimpleNamespace(optimize_mu=True,
                                    marg_modes=args.num_modes,
                                    batch_size=args.num_modes,
                                    use_tanh=args.use_tanh,
                                    init_std=1e-2,
                                    cov_diagonal=args.cov_diagonal,
                                    cov_off_diagonal=args.cov_off_diagonal,
                                    average=args.average,
                                    #ff_residual_connection=False,
                                    #ff_activation='tanh',
                                    #ff_layer_norm=False,
                                    #ff_layers=1,
                                    )
    #print(f"dim = {args.dimension}")

    class_ = KNIFE
    
    if hasattr(args, 'class_'):
        if args.class_ == 'MargKernel':
            class_ = MargKernel
    
    G_adaptive = class_(args_adaptive,
                            args.dimension,
                            None,
                            init_samples=torch.tensor(kernel_samples).to(args.device),
                            )
    G_adaptive = G_adaptive.to(args.device)

    opt = torch.optim.Adam(G_adaptive.parameters(), lr=args.lr)

    train_losses = []
    test_losses = []

    for i_epoch in range(args.epochs):
        #logger.info(f'Epoch {i_epoch+1!s}/{args.epochs!s} starting ...')
        data_iter = iter(dataloader)
        sum = np.array(0.0)

        with (tqdm(total = len(dataloader)) if print_log else nullcontext()) as progress_bar:
            for i, x in enumerate(data_iter):
                if print_log:
                    progress_bar.update()

                opt.zero_grad()

                x = x.to(args.device)

                loss_adaptive = G_adaptive(x)
                loss_adaptive.backward()
                loss_np = loss_adaptive.detach().cpu().numpy()
                sum += loss_np
                avg_train_loss = sum / (i+1)#Not entirely when having a final batch of different size
                if print_log:
                    progress_bar.set_description(f"Train loss: {avg_train_loss:.3f}")

                opt.step()

            train_losses.append(avg_train_loss)

            sum = np.array(0.0)
            for x in iter(test_loader):
                x = x.to(args.device)
                loss_np = G_adaptive(x).detach().cpu().numpy()
                sum += x.shape[0] * loss_np

            avg_test_loss = sum / len(test_loader.dataset)

            test_losses.append(avg_test_loss)

            G_adaptive.cross_entropy = avg_test_loss # dumb hack to get the test loss in the end

            if print_log:
                progress_bar.set_description(f"{i_epoch + 1}: Train loss: {avg_train_loss:.3f} \t Test loss: {avg_test_loss:.3f}")
                progress_bar.refresh()

    if print_log:
        print("Finished kernel-fitting.")

    if return_history:
        return G_adaptive, train_losses, test_losses
    else:
        return G_adaptive
