from numbers import Number

import numpy as np
import torch
from torch import distributions as dist
from torch import nn
from torch.nn import functional as F


def weights_init(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight.data)


class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim, n_layers, activation='none', slope=.1, device='cpu'):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.n_layers = n_layers
        self.device = device
        if isinstance(hidden_dim, Number):
            self.hidden_dim = [hidden_dim] * (self.n_layers - 1)
        elif isinstance(hidden_dim, list):
            self.hidden_dim = hidden_dim
        else:
            raise ValueError('Wrong argument type for hidden_dim: {}'.format(hidden_dim))

        if isinstance(activation, str):
            self.activation = [activation] * (self.n_layers - 1)
        elif isinstance(activation, list):
            self.hidden_dim = activation
        else:
            raise ValueError('Wrong argument type for activation: {}'.format(activation))

        self._act_f = []
        for act in self.activation:
            if act == 'lrelu':
                self._act_f.append(lambda x: F.leaky_relu(x, negative_slope=slope))
            elif act == 'xtanh':
                self._act_f.append(lambda x: self.xtanh(x, alpha=slope))
            elif act == 'sigmoid':
                self._act_f.append(F.sigmoid)
            elif act == 'none':
                self._act_f.append(lambda x: x)
            else:
                ValueError('Incorrect activation: {}'.format(act))

        if self.n_layers == 1:
            _fc_list = [nn.Linear(self.input_dim, self.output_dim)]
        else:
            _fc_list = [nn.Linear(self.input_dim, self.hidden_dim[0])]
            for i in range(1, self.n_layers - 1):
                _fc_list.append(nn.Linear(self.hidden_dim[i - 1], self.hidden_dim[i]))
            _fc_list.append(nn.Linear(self.hidden_dim[self.n_layers - 2], self.output_dim))
        self.fc = nn.ModuleList(_fc_list)
        self.to(self.device)

    @staticmethod
    def xtanh(x, alpha=.1):
        """tanh function plus an additional linear term"""
        return x.tanh() + alpha * x

    def forward(self, x):
        h = x
        for c in range(self.n_layers):
            if c == self.n_layers - 1:
                h = self.fc[c](h)
            else:
                h = self._act_f[c](self.fc[c](h))
        return h


class Dist:
    def __init__(self):
        pass

    def sample(self, *args):
        pass

    def log_pdf(self, *args, **kwargs):
        pass


class Normal(Dist):
    def __init__(self, device='cpu'):
        super().__init__()
        self.device = device
        self.c = 2 * np.pi * torch.ones(1).to(self.device)
        self._dist = dist.normal.Normal(torch.zeros(1).to(self.device), torch.ones(1).to(self.device))
        self.name = 'gauss'

    def sample(self, mu, v):
        eps = self._dist.sample(mu.size()).squeeze()
        scaled = eps.mul(v.sqrt())
        return scaled.add(mu)

    def log_pdf(self, x, mu, v, reduce=True, param_shape=None):
        """compute the log-pdf of a normal distribution with diagonal covariance"""
        if param_shape is not None:
            mu, v = mu.view(param_shape), v.view(param_shape)
        lpdf = -0.5 * (torch.log(self.c) + v.log() + (x - mu).pow(2).div(v))
        if reduce:
            return lpdf.sum(dim=-1)
        else:
            return lpdf

    def log_pdf_full(self, x, mu, v):
        """
        compute the log-pdf of a normal distribution with full covariance
        v is a batch of "pseudo sqrt" of covariance matrices of shape (batch_size, d_latent, d_latent)
        mu is batch of means of shape (batch_size, d_latent)
        """
        batch_size, d = mu.size()
        cov = torch.einsum('bik,bjk->bij', v, v)  # compute batch cov from its "pseudo sqrt"
        assert cov.size() == (batch_size, d, d)
        inv_cov = torch.inverse(cov)  # works on batches
        c = d * torch.log(self.c)
        # matrix log det doesn't work on batches!
        _, logabsdets = self._batch_slogdet(cov)
        xmu = x - mu
        return -0.5 * (c + logabsdets + torch.einsum('bi,bij,bj->b', [xmu, inv_cov, xmu]))

    def _batch_slogdet(self, cov_batch: torch.Tensor):
        """
        compute the log of the absolute value of determinants for a batch of 2D matrices. Uses torch.slogdet
        this implementation is just a for loop, but that is what's suggested in torch forums
        gpu compatible
        """
        batch_size = cov_batch.size(0)
        signs = torch.empty(batch_size, requires_grad=False).to(self.device)
        logabsdets = torch.empty(batch_size, requires_grad=False).to(self.device)
        for i, cov in enumerate(cov_batch):
            signs[i], logabsdets[i] = torch.slogdet(cov)
        return signs, logabsdets


class Laplace(Dist):
    def __init__(self, device='cpu'):
        super().__init__()
        self.device = device
        self._dist = dist.laplace.Laplace(torch.zeros(1).to(self.device), torch.ones(1).to(self.device) / np.sqrt(2))
        self.name = 'laplace'

    def sample(self, mu, b):
        eps = self._dist.sample(mu.size())
        scaled = eps.mul(b)
        return scaled.add(mu)

    def log_pdf(self, x, mu, b, reduce=True, param_shape=None):
        """compute the log-pdf of a laplace distribution with diagonal covariance"""
        if param_shape is not None:
            mu, b = mu.view(param_shape), b.view(param_shape)
        lpdf = -torch.log(2 * b) - (x - mu).abs().div(b)
        if reduce:
            return lpdf.sum(dim=-1)
        else:
            return lpdf


class GaussianMLP(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim, n_layers, activation, slope, device, fixed_mean=None,
                 fixed_var=None):
        super().__init__()
        self.distribution = Normal(device=device)
        if fixed_mean is None:
            self.mean = MLP(input_dim, output_dim, hidden_dim, n_layers, activation=activation, slope=slope,
                            device=device)
        else:
            self.mean = lambda x: fixed_mean * torch.ones(1).to(device)
        if fixed_var is None:
            self.log_var = MLP(input_dim, output_dim, hidden_dim, n_layers, activation=activation, slope=slope,
                               device=device)
        else:
            self.log_var = lambda x: np.log(fixed_var) * torch.ones(1).to(device)

    def sample(self, *params):
        return self.distribution.sample(*params)

    def log_pdf(self, x, *params, **kwargs):
        return self.distribution.log_pdf(x, *params, **kwargs)


    def forward(self, *input):
        if len(input) > 1:
            x = torch.cat(input, dim=1)
        else:
            x = input[0]
        return self.mean(x), self.log_var(x).exp()


class ModularIVAE(nn.Module):
    def __init__(self, latent_dim, data_dim, aux_dim, prior=None, decoder=None, encoder=None,
                 n_layers=3, hidden_dim=50, activation='lrelu', slope=.1, device='cpu', anneal=False):
        super().__init__()

        self.data_dim = data_dim
        self.latent_dim = latent_dim
        self.aux_dim = aux_dim
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.activation = activation
        self.slope = slope
        self.anneal_params = anneal

        if prior is None:
            self.prior = GaussianMLP(aux_dim, latent_dim, hidden_dim, n_layers, activation=activation, slope=slope,
                                     device=device, fixed_mean=0)
        else:
            self.prior = prior

        if decoder is None:
            self.decoder = GaussianMLP(latent_dim, data_dim, hidden_dim, n_layers, activation=activation, slope=slope,
                                       device=device, fixed_var=.01)
        else:
            self.decoder = decoder

        if encoder is None:
            self.encoder = GaussianMLP(data_dim + aux_dim, latent_dim, hidden_dim, n_layers, activation=activation,
                                       slope=slope, device=device)
        else:
            self.encoder = encoder

        self.apply(weights_init)

        self._training_hyperparams = [1., 1., 1., 1., 1]

    def forward(self, x, u):
        encoder_params = self.encoder(x, u)
        z = self.encoder.sample(*encoder_params)
        decoder_params = self.decoder(z)
        prior_params = self.prior(u)
        return decoder_params, encoder_params, prior_params, z

    def elbo(self, x, u):
        decoder_params, encoder_params, prior_params, z = self.forward(x, u)
        log_px_z = self.decoder.log_pdf(x, *decoder_params)
        log_qz_xu = self.encoder.log_pdf(z, *encoder_params)
        log_pz_u = self.prior.log_pdf(z, *prior_params)

        if self.anneal_params:
            a, b, c, d, N = self._training_hyperparams
            M = z.size(0)
            log_qz_tmp = self.encoder.log_pdf(z.view(M, 1, self.latent_dim), encoder_params, reduce=False,
                                              param_shape=(1, M, self.latent_dim))
            log_qz = torch.logsumexp(log_qz_tmp.sum(dim=-1), dim=1, keepdim=False) - np.log(M * N)
            log_qz_i = (torch.logsumexp(log_qz_tmp, dim=1, keepdim=False) - np.log(M * N)).sum(dim=-1)

            return (a * log_px_z - b * (log_qz_xu - log_qz) - c * (log_qz - log_qz_i) - d * (
                    log_qz_i - log_pz_u)).mean(), z
        else:
            return (log_px_z + log_pz_u - log_qz_xu).mean(), z

    def anneal(self, N, max_iter, it):
        thr = int(max_iter / 1.6)
        a = 0.5 / self.decoder.log_var(0).exp().item()
        self._training_hyperparams[-1] = N
        self._training_hyperparams[0] = min(2 * a, a + a * it / thr)
        self._training_hyperparams[1] = max(1, a * .3 * (1 - it / thr))
        self._training_hyperparams[2] = min(1, it / thr)
        self._training_hyperparams[3] = max(1, a * .5 * (1 - it / thr))
        if it > thr:
            self.anneal_params = False


class iVAE(nn.Module):
    def __init__(self, latent_dim, data_dim, aux_dim, prior=None, decoder=None, encoder=None,
                 n_layers=3, hidden_dim=50, activation='lrelu', slope=.1, device='cpu', anneal=False):
        super().__init__()

        self.data_dim = data_dim
        self.latent_dim = latent_dim
        self.aux_dim = aux_dim
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.activation = activation
        self.slope = slope
        self.anneal_params = anneal

        if prior is None:
            self.prior_dist = Normal(device=device)
        else:
            self.prior_dist = prior

        if decoder is None:
            self.decoder_dist = Normal(device=device)
        else:
            self.decoder_dist = decoder

        if encoder is None:
            self.encoder_dist = Normal(device=device)
        else:
            self.encoder_dist = encoder

        # prior_params
        self.prior_mean = torch.zeros(1).to(device)
        self.logl = MLP(aux_dim, latent_dim, hidden_dim, n_layers, activation=activation, slope=slope, device=device)
        # decoder params
        self.f = MLP(latent_dim, data_dim, hidden_dim, n_layers, activation=activation, slope=slope, device=device)
        self.decoder_var = .01 * torch.ones(1).to(device)
        # encoder params
        self.g = MLP(data_dim + aux_dim, latent_dim, hidden_dim, n_layers, activation=activation, slope=slope,
                     device=device)
        self.logv = MLP(data_dim + aux_dim, latent_dim, hidden_dim, n_layers, activation=activation, slope=slope,
                        device=device)

        self.apply(weights_init)

        self._training_hyperparams = [1., 1., 1., 1., 1]

    def encoder_params(self, x, u):
        xu = torch.cat((x, u), 1)
        g = self.g(xu)
        logv = self.logv(xu)
        return g, logv.exp()

    def decoder_params(self, s):
        f = self.f(s)
        return f, self.decoder_var

    def prior_params(self, u):
        logl = self.logl(u)
        return self.prior_mean, logl.exp()

    def forward(self, x, u):
        prior_params = self.prior_params(u)
        encoder_params = self.encoder_params(x, u)
        z = self.encoder_dist.sample(*encoder_params)
        decoder_params = self.decoder_params(z)
        return decoder_params, encoder_params, z, prior_params

    def elbo(self, x, u):
        decoder_params, (g, v), z, prior_params = self.forward(x, u)
        log_px_z = self.decoder_dist.log_pdf(x, *decoder_params)
        log_qz_xu = self.encoder_dist.log_pdf(z, g, v)
        log_pz_u = self.prior_dist.log_pdf(z, *prior_params)

        if self.anneal_params:
            a, b, c, d, N = self._training_hyperparams
            M = z.size(0)
            log_qz_tmp = self.encoder_dist.log_pdf(z.view(M, 1, self.latent_dim), g.view(1, M, self.latent_dim),
                                                   v.view(1, M, self.latent_dim), reduce=False)
            log_qz = torch.logsumexp(log_qz_tmp.sum(dim=-1), dim=1, keepdim=False) - np.log(M * N)
            log_qz_i = (torch.logsumexp(log_qz_tmp, dim=1, keepdim=False) - np.log(M * N)).sum(dim=-1)

            return (a * log_px_z - b * (log_qz_xu - log_qz) - c * (log_qz - log_qz_i) - d * (
                    log_qz_i - log_pz_u)).mean(), z

        else:
            return (log_px_z + log_pz_u - log_qz_xu).mean(), z

    def anneal(self, N, max_iter, it):
        thr = int(max_iter / 1.6)
        a = 0.5 / self.decoder_var.item()
        self._training_hyperparams[-1] = N
        self._training_hyperparams[0] = min(2 * a, a + a * it / thr)
        self._training_hyperparams[1] = max(1, a * .3 * (1 - it / thr))
        self._training_hyperparams[2] = min(1, it / thr)
        self._training_hyperparams[3] = max(1, a * .5 * (1 - it / thr))
        if it > thr:
            self.anneal_params = False


# MNIST MODELS

class ConvolutionalVAEforMNIST(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()

        self.latent_dim = latent_dim
        self.encoder = nn.Sequential(
            # input is mnist image: 1x28x28
            nn.Conv2d(1, 32, 4, 2, 1),  # 32x14x14
            nn.BatchNorm2d(32),  # 32x14x14
            nn.ReLU(inplace=True),  # 32x14x14
            nn.Conv2d(32, 128, 4, 2, 1),  # 128x7x7
            nn.BatchNorm2d(128),  # 128x7x7
            nn.ReLU(inplace=True),  # 128x7x7
            nn.Conv2d(128, 512, 7, 1, 0),  # 512x1x1
            nn.BatchNorm2d(512),  # 512x1x1
            nn.ReLU(inplace=True),  # 512x1x1
            nn.Conv2d(512, 200, 1, 1, 0),  # 200x1x1
        )
        self.fc1 = nn.Linear(200, latent_dim)
        self.fc2 = nn.Linear(200, latent_dim)

        self.decoder = nn.Sequential(
            # input: latent_dim x 1 x 1
            nn.ConvTranspose2d(latent_dim, 512, 1, 1, 0),  # 512x1x1
            nn.BatchNorm2d(512),  # 512x1x1
            nn.ReLU(inplace=True),  # 512x1x1
            nn.ConvTranspose2d(512, 128, 7, 1, 0),  # 128x7x7
            nn.BatchNorm2d(128),  # 128x7x7
            nn.ReLU(inplace=True),  # 128x7x7
            nn.ConvTranspose2d(128, 32, 4, 2, 1),  # 32x14x14
            nn.BatchNorm2d(32),  # 32x14x14
            nn.ReLU(inplace=True),  # 32x14x14
            nn.ConvTranspose2d(32, 1, 4, 2, 1)  # 1x28x28
        )

    def encode(self, x):
        h = self.encoder(x.view(-1, 1, 28, 28)).squeeze()
        return self.fc1(F.relu(h)), self.fc2(F.relu(h))

    def decode(self, z):
        h = self.decoder(z.view(z.size(0), z.size(1), 1, 1))
        return torch.sigmoid(h.view(-1, 28 * 28))

    @staticmethod
    def reparameterize(mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        mu, logv = self.encode(x)
        z = self.reparameterize(mu, logv)
        f = self.decode(z)
        return f, mu, logv


class VAEforMNIST(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()

        self.latent_dim = latent_dim

        self.fc1 = nn.Linear(784, 400)
        self.fc21 = nn.Linear(400, latent_dim)
        self.fc22 = nn.Linear(400, latent_dim)
        self.fc3 = nn.Linear(latent_dim, 400)
        self.fc4 = nn.Linear(400, 784)

    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)

    @staticmethod
    def reparameterize(mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h3 = F.relu(self.fc3(z))
        return torch.sigmoid(self.fc4(h3))

    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, 784))
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar


class ConvolutionalIVAEforMNIST(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()

        self.latent_dim = latent_dim

        # encoder
        self.encoder = nn.Sequential(
            # input is mnist image: 1x28x28
            nn.Conv2d(1, 32, 4, 2, 1),  # 32x14x14
            nn.BatchNorm2d(32),  # 32x14x14
            nn.ReLU(inplace=True),  # 32x14x14
            nn.Conv2d(32, 128, 4, 2, 1),  # 128x7x7
            nn.BatchNorm2d(128),  # 128x7x7
            nn.ReLU(inplace=True),  # 128x7x7
            nn.Conv2d(128, 512, 7, 1, 0),  # 512x1x1
            nn.BatchNorm2d(512),  # 512x1x1
            nn.ReLU(inplace=True),  # 512x1x1
            nn.Conv2d(512, 200, 1, 1, 0),  # 200x1x1
        )
        self.fc1 = nn.Linear(200, latent_dim)
        self.fc2 = nn.Linear(200, latent_dim)

        # decoder
        self.decoder = nn.Sequential(
            # input: latent_dim x 1 x 1
            nn.ConvTranspose2d(latent_dim, 512, 1, 1, 0),  # 512x1x1
            nn.BatchNorm2d(512),  # 512x1x1
            nn.ReLU(inplace=True),  # 512x1x1
            nn.ConvTranspose2d(512, 128, 7, 1, 0),  # 128x7x7
            nn.BatchNorm2d(128),  # 128x7x7
            nn.ReLU(inplace=True),  # 128x7x7
            nn.ConvTranspose2d(128, 32, 4, 2, 1),  # 32x14x14
            nn.BatchNorm2d(32),  # 32x14x14
            nn.ReLU(inplace=True),  # 32x14x14
            nn.ConvTranspose2d(32, 1, 4, 2, 1)  # 1x28x28
        )

        # prior
        self.l1 = nn.Linear(10, 200)
        self.l21 = nn.Linear(200, latent_dim)
        self.l22 = nn.Linear(200, latent_dim)

    def encode(self, x):
        h = self.encoder(x.view(-1, 1, 28, 28)).squeeze()
        return self.fc1(F.relu(h)), self.fc2(F.relu(h))

    def decode(self, z):
        h = self.decoder(z.view(z.size(0), z.size(1), 1, 1))
        return torch.sigmoid(h.view(-1, 28 * 28))

    def prior(self, y):
        h2 = F.relu(self.l1(y))
        # h2 = self.l1(y)
        return self.l21(h2), self.l22(h2)

    @staticmethod
    def reparameterize(mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x, y):
        mu, logv = self.encode(x)
        mup, logl = self.prior(y)
        z = self.reparameterize(mu, logv)
        f = self.decode(z)
        return f, mu, logv, mup, logl


class iVAEforMNIST(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()

        self.latent_dim = latent_dim

        self.fc1 = nn.Linear(784, 400)
        self.fc21 = nn.Linear(400, latent_dim)
        self.fc22 = nn.Linear(400, latent_dim)
        self.fc3 = nn.Linear(latent_dim, 400)
        self.fc4 = nn.Linear(400, 784)

        hidden_dim = 200
        self.l1 = nn.Linear(10, hidden_dim)
        self.l21 = nn.Linear(hidden_dim, latent_dim)
        self.l22 = nn.Linear(hidden_dim, latent_dim)

    def encode(self, x):
        h1 = F.leaky_relu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)

    def prior(self, y):
        h2 = F.relu(self.l1(y))
        # h2 = self.l1(y)
        return self.l21(h2), self.l22(h2)

    def decode(self, z):
        h3 = F.leaky_relu(self.fc3(z))
        return torch.sigmoid(self.fc4(h3))

    @staticmethod
    def reparameterize(mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x, y):
        mu, logvar = self.encode(x.view(-1, 784))
        mup, logl = self.prior(y)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar, mup, logl
