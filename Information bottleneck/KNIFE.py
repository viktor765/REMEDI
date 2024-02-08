import numpy as np
import torch
import torch.nn as nn
import torch.distributions as D

from types import SimpleNamespace
from torch.utils.data import DataLoader
from tqdm import tqdm
from contextlib import nullcontext

class MargKernel(nn.Module):
    def __init__(self, args, zc_dim, zd_dim, init_samples=None):

        self.lr = args.lr       ## modified
        self.epochs = args.epochs ## modified
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
        assert x.shape[1] == self.d, 'x has to have shape [N, d]'
        assert x.dtype == self.dtype, f'x has to be of type {self.dtype}'

        x = x[:, None, :]
        w = torch.log_softmax(self.weigh, dim=1)
        y = x - self.means
        #print(f"exp term: {y}",flush=True)
        logvar = self.logvar
        if self.use_tanh:
            logvar = logvar.tanh()
        var = logvar.exp()
        y = y * var
        if self.tri is not None:
            y = y + torch.squeeze(torch.matmul(torch.tril(self.tri, diagonal=-1), y[:, :, :, None]), 3)
        y = torch.sum(y ** 2, dim=2)


        #print(f"var term: {torch.sum(torch.log(torch.abs(var) + 1e-8), dim=-1)}",flush=True)
        y = -y / 2 + torch.sum(torch.log(torch.abs(var) + 1e-8), dim=-1) + w
        
        
        y = torch.logsumexp(y, dim=-1)            

        # clone y to tmp
        tmp = y.clone()
        
        # rows of tmp with sum of exp values == 0
        tmp[tmp == float('-inf')] = torch.tensor([-1e5])
        y = tmp
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
        #print(L.shape,flush=True)
        I = D.Categorical(weights).sample((n,))
        #print(I.shape,flush=True)
        mu_I = mu[I]
        #print(mu_I.shape,flush=True)
        L_I = L.squeeze(0)[I]
        #print(L_I.shape,flush=True)

        d = D.MultivariateNormal(torch.zeros_like(mu_I).cuda(), torch.eye(mu_I.shape[1]).cuda())
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

    def weight_averaged_sample_loss(self, T, n):
        if type(n) is tuple:
            n = n[0]

        weights = torch.softmax(self.weigh, dim=1).squeeze(0)
        mu = self.means
        var = self.logvar.exp()
        tri = torch.tril(self.tri, diagonal=-1)

        V = torch.diag_embed(var)
        L = (V + tri @ V).squeeze(0)#Cholesky factor of precision matrix
            
        d = D.MultivariateNormal(torch.zeros_like(mu), torch.eye(mu.shape[1]))

        z = d.sample((n,))
        # print(z.shape)
        z1 = torch.linalg.solve_triangular(L, z.unsqueeze(-1), upper=False).squeeze(-1) 
        samples = z1 + mu
        # print(z1.shape)
        # print(T(samples).shape)
        # print(torch.linalg.solve_triangular(L, z.unsqueeze(-1), upper=False).shape)

        return T(samples).squeeze(-1) @ weights
    
def fit_kernel(args,optimizer, print_log):
    dataloader = DataLoader(args.train_samples, batch_size=args.batchsize, shuffle=args.shuffle, drop_last=False)
    #test_loader = DataLoader(args.test_samples, batch_size=args.batchsize, shuffle=False, drop_last=False)

    kernel_samples = args.train_samples[np.random.choice(len(args.train_samples), args.num_modes, replace=False)]

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

    opt = optimizer if optimizer is not None else torch.optim.Adam(G_adaptive.parameters(), lr=args.lr)

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

            sum = np.array(0.0)
            '''for x in iter(test_loader):
                x = x.to(args.device)
                loss_np = G_adaptive(x).detach().cpu().numpy()
                sum += x.shape[0] * loss_np

            avg_test_loss = sum / len(test_loader.dataset)
            '''
            if print_log:
                progress_bar.set_description(f"{i_epoch + 1}: Train loss: {avg_train_loss:.3f} ")#\t Test loss: {avg_test_loss:.3f}")
                progress_bar.refresh()

    if print_log:
        print("Finished kernel-fitting.")

    return G_adaptive
