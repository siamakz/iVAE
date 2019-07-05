"""
Script for generating piecew-wise stationary data.

Each component of the independent latents is comprised of `ns` segments, and each segment has different parameters.\
Each segment has `nps` data points 9measurements).

The latent components are then mixed by an MLP into observations (not necessarily of the same dimension.
It is possible to add noise to the observations
"""

import math
import os

import numpy as np
import scipy
import torch
from scipy.stats import hypsecant
from torch.utils.data import Dataset


def to_one_hot(x, m=None):
    if type(x) is not list:
        x = [x]
    if m is None:
        ml = []
        for xi in x:
            ml += [xi.max() + 1]
        m = max(ml)
    dtp = x[0].dtype
    xoh = []
    for i, xi in enumerate(x):
        xoh += [np.zeros((xi.size, int(m)), dtype=dtp)]
        xoh[i][np.arange(xi.size), xi.astype(np.int)] = 1
    return xoh


def lrelu(x, neg_slope):
    """
    Leaky ReLU activation function
    @param x: input array
    @param neg_slope: slope for negative values
    @return:
        out: output rectified array
    """

    def _lrelu_1d(_x, _neg_slope):
        """
        one dimensional implementation of leaky ReLU
        """
        if _x > 0:
            return _x
        else:
            return _x * _neg_slope

    leaky1d = np.vectorize(_lrelu_1d)
    assert neg_slope > 0  # must be positive
    return leaky1d(x, neg_slope)


def sigmoid(x):
    """
    Sigmoid activation function
    @param x: input array
    @return:
        out: output array
    """
    return 1 / (1 + np.exp(-x))


def generate_mixing_matrix(d_sources: int, d_data=None, lin_type='uniform', cond_threshold=25, n_iter_4_cond=None,
                           dtype=np.float32):
    """
    Generate square linear mixing matrix
    @param d_sources: dimension of the latent sources
    @param d_data: dimension of the mixed data
    @param lin_type: specifies the type of matrix entries; either `uniform` or `orthogonal`.
    @param cond_threshold: higher bound on the condition number of the matrix to ensure well-conditioned problem
    @param n_iter_4_cond: or instead, number of iteration to compute condition threshold of the mixing matrix.
        cond_threshold is ignored in this case/
    @param dtype: data type for data
    @return:
        A: mixing matrix
    @rtype: np.ndarray
    """
    if d_data is None:
        d_data = d_sources

    if lin_type == 'orthogonal':
        A = (np.linalg.qr(np.random.uniform(-1, 1, (d_sources, d_data)))[0]).astype(dtype)

    elif lin_type == 'uniform':
        if n_iter_4_cond is None:
            cond_thresh = cond_threshold
        else:
            cond_list = []
            for _ in range(int(n_iter_4_cond)):
                A = np.random.uniform(-1, 1, (d_sources, d_data)).astype(dtype)
                for i in range(d_data):
                    A[:, i] /= np.sqrt((A[:, i] ** 2).sum())
                cond_list.append(np.linalg.cond(A))

            cond_thresh = np.percentile(cond_list, 25)  # only accept those below 25% percentile

        A = (np.random.uniform(0, 2, (d_sources, d_data)) - 1).astype(dtype)
        for i in range(d_data):
            A[:, i] /= np.sqrt((A[:, i] ** 2).sum())

        while np.linalg.cond(A) > cond_thresh:
            # generate a new A matrix!
            A = (np.random.uniform(0, 2, (d_sources, d_data)) - 1).astype(dtype)
            for i in range(d_data):
                A[:, i] /= np.sqrt((A[:, i] ** 2).sum())
    else:
        raise ValueError('incorrect method')
    return A


def generate_nonstationary_sources(n_per_seg: int, n_seg: int, d: int, prior='lap', var_bounds=np.array([0.5, 3]),
                                   dtype=np.float32, uncentered=False):
    """
    Generate source signal following a TCL distribution. Within each segment, sources are independent.
    The distribution withing each segment is given by the keyword `dist`
    @param n_per_seg: number of points per segment
    @param n_seg: number of segments
    @param d: dimension of the sources same as data
    @param prior: distribution of the sources. can be `lap` for Laplace , `hs` for Hypersecant or `gauss` for Gaussian
    @param var_bounds: optional, upper and lower bounds for the modulation parameter
    @param dtype: data type for data
    @param bool uncentered: True to generate uncentered data
    @return:
        sources: output source array of shape (n, d)
        labels: label for each point; the label is the component
        m: mean of each component
        L: modulation parameter of each component
    @rtype: (np.ndarray, np.ndarray, np.ndarray, np.ndarray)
    """
    var_lb = var_bounds[0]
    var_ub = var_bounds[1]
    n = n_per_seg * n_seg

    L = np.random.uniform(var_lb, var_ub, (n_seg, d))
    if uncentered:
        m = np.random.uniform(-5, 5, (n_seg, d))
    else:
        m = np.zeros((n_seg, d))

    labels = np.zeros(n, dtype=dtype)
    if prior == 'lap':
        sources = np.random.laplace(0, 1 / np.sqrt(2), (n, d)).astype(dtype)
    elif prior == 'hs':
        sources = scipy.stats.hypsecant.rvs(0, 1, (n, d)).astype(dtype)
    elif prior == 'gauss':
        sources = np.random.randn(n, d).astype(dtype)
    else:
        raise ValueError('incorrect dist')

    for seg in range(n_seg):
        segID = range(n_per_seg * seg, n_per_seg * (seg + 1))
        sources[segID] *= L[seg]
        sources[segID] += m[seg]
        labels[segID] = seg

    return sources, labels, m, L


def generate_data(n_per_seg, n_seg, d_sources, d_data=None, n_layers=3, prior='lap', activation='lrelu', batch_size=250,
                  seed=10, slope=.1, var_bounds=np.array([0.5, 3]), lin_type='uniform', n_iter_4_cond=1e4,
                  dtype=np.float32, uncentered=False, noisy=0):
    """
    Generate artificial data with arbitrary mixing
    @param int n_per_seg: number of observations per segment
    @param int n_seg: number of segments
    @param int d_sources: dimension of the latent sources
    @param int or None d_data: dimension of the data
    @param int n_layers: number of layers in the mixing MLP
    @param str activation: activation function for the mixing MLP; can be `none, `lrelu`, `xtanh` or `sigmoid`
    @param str prior: prior distribution of the sources; can be `lap` for Laplace or `hs` for Hypersecant
    @param int batch_size: batch size if data is to be returned as batches. 0 for a single batch of size n
    @param int seed: random seed
    @param var_bounds: upper and lower bounds for the modulation parameter
    @param float slope: slope parameter for `lrelu` or `xtanh`
    @param str lin_type: specifies the type of matrix entries; can be `uniform` or `orthogonal`
    @param int n_iter_4_cond: number of iteration to compute condition threshold of the mixing matrix
    @param dtype: data type for data
    @param bool uncentered: True to generate uncentered data
    @param float noisy: if non-zero, controls the level of noise added to observations

    @return:
        tuple of batches of generated (sources, data, auxiliary variables, mean, variance)
    @rtype: tuple

    """
    if seed is not None:
        np.random.seed(seed)

    if d_data is None:
        d_data = d_sources

    # sources
    sources, labels, m, L = generate_nonstationary_sources(n_per_seg, n_seg, d_sources, prior=prior,
                                                           var_bounds=var_bounds,
                                                           dtype=dtype, uncentered=uncentered)
    n = n_per_seg * n_seg

    # non linearity
    if activation == 'lrelu':
        act_f = lambda x: lrelu(x, slope).astype(dtype)
    elif activation == 'sigmoid':
        act_f = sigmoid
    elif activation == 'xtanh':
        act_f = lambda x: np.tanh(x) + slope * x
    elif activation == 'none':
        act_f = lambda x: x
    else:
        raise ValueError('incorrect non linearity: {}'.format(activation))

    # Mixing time!
    assert n_layers > 1  # suppose we always have at least 2 layers. The last layer doesn't have a non-linearity
    A = generate_mixing_matrix(d_sources, d_data, lin_type=lin_type, n_iter_4_cond=n_iter_4_cond, dtype=dtype)
    X = act_f(np.dot(sources, A))
    if d_sources != d_data:
        B = generate_mixing_matrix(d_data, lin_type=lin_type, n_iter_4_cond=n_iter_4_cond, dtype=dtype)
    else:
        B = A
    for nl in range(1, n_layers):
        if nl == n_layers - 1:
            X = np.dot(X, B)
        else:
            X = act_f(np.dot(X, B))

    # add noise:
    if noisy:
        X += noisy * np.random.randn(*X.shape)

    # always return batches (as a list), even if number of batches is one,
    if not batch_size:
        return [sources], [X], to_one_hot([labels], m=n_seg), m, L
    else:
        idx = np.random.permutation(n)
        Xb, Sb, Ub = [], [], []
        n_batches = int(n / batch_size)
        for c in range(n_batches):
            Sb += [sources[idx][c * batch_size:(c + 1) * batch_size]]
            Xb += [X[idx][c * batch_size:(c + 1) * batch_size]]
            Ub += [labels[idx][c * batch_size:(c + 1) * batch_size]]
        return Sb, Xb, to_one_hot(Ub, m=n_seg), m, L


def save_data(path, *args, **kwargs):
    kwargs['batch_size'] = 0  # leave batch creation to torch DataLoader
    Sb, Xb, Ub, m, L = generate_data(*args, **kwargs)
    Sb, Xb, Ub = Sb[0], Xb[0], Ub[0]
    print('Creating dataset {} ...'.format(path))
    dir_path = '/'.join(path.split('/')[:-1])
    if not os.path.exists(dir_path):
        os.makedirs('/'.join(path.split('/')[:-1]))
    np.savez_compressed(path, s=Sb, x=Xb, u=Ub, m=m, L=L)
    print(' ... done')


class SyntheticDataset(Dataset):
    def __init__(self, path, device='cpu'):
        self.device = device
        self.path = path
        try:
            data = np.load(path)
        except:
            # error occured because many scripts were attempting to create it at same time.
            # one solution would be to wait and retry, the other would be to make sure
            # datasets are all created already.
            pass
        self.data = data
        self.s = torch.from_numpy(data['s']).to(self.device)
        self.x = torch.from_numpy(data['x']).to(self.device)
        self.u = torch.from_numpy(data['u']).to(self.device)
        self.L = data['L']
        self.M = data['m']
        self.len = self.x.shape[0]
        self.latent_dim = self.s.shape[1]
        self.aux_dim = self.u.shape[1]
        self.data_dim = self.x.shape[1]
        self.nps = int(self.len / self.aux_dim)
        print('data loaded on {}'.format(self.x.device))

    def get_dims(self):
        return self.data_dim, self.latent_dim, self.aux_dim

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        return self.x[index], self.u[index], self.s[index]

    def get_metadata(self):
        return {'path': self.path,
                'nps': self.nps,
                'ns': self.aux_dim,
                'n': self.len,
                'latent_dim': self.latent_dim,
                'data_dim': self.data_dim,
                'aux_dim': self.aux_dim,
                }


class DataLoaderGPU:
    """
    A custom data loader on GPU.
    """
    def __init__(self, path, batch_size, shuffle=True):
        self.device = torch.device('cuda')
        self.path = path
        data = np.load(path)
        self.data = data
        print('data loaded on {}'.format(self.device))
        self.s = torch.from_numpy(data['s']).to(self.device)
        self.x = torch.from_numpy(data['x']).to(self.device)
        self.u = torch.from_numpy(data['u']).to(self.device)
        self.dataset_len = self.x.shape[0]
        self.latent_dim = self.s.shape[1]
        self.aux_dim = self.u.shape[1]
        self.data_dim = self.x.shape[1]
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.len = math.ceil(self.dataset_len / self.batch_size)
        self.nps = int(self.dataset_len / self.aux_dim)
        if self.shuffle:
            self.idx = np.random.permutation(self.dataset_len)
        else:
            self.idx = np.arange(self.dataset_len)

    def get_dims(self):
        return self.data_dim, self.latent_dim, self.aux_dim

    def __len__(self):
        return self.len

    def __iter__(self):
        for b in range(self.len):
            idx = self.idx[self.batch_size * b:self.batch_size * (b + 1)]
            yield self.x[idx], self.u[idx], self.s[idx]

    def get_metadata(self):
        return {'path': self.path,
                'nps': self.nps,
                'ns': self.aux_dim,
                'n': self.dataset_len,
                'latent_dim': self.latent_dim,
                'data_dim': self.data_dim,
                'aux_dim': self.aux_dim,
                }


def create_if_not_exist_dataset(root='data/', nps=1000, ns=40, dl=2, dd=4, nl=3, s=1, p='gauss', a='xtanh',
                                uncentered=False, noisy=False, arg_str=None):
    """
    Create a dataset if it doesn't exist.
    This is useful as a setup step when running multiple jobs in parallel, to avoid having many scripts attempting
    to create the dataset when non-existent.
    This is called in `cmd_utils.create_dataset_before`
    """
    if arg_str is not None:
        # overwrites all other arg values
        # arg_str should be of this form: nps_ns_dl_dd_nl_s_p_a_u_n
        arg_list = arg_str.split('_')
        assert len(arg_list) == 10
        nps, ns, dl, dd, nl = map(int, arg_list[0:5])
        p, a = arg_list[6:8]
        if arg_list[5] == 'n':
            s = None
        else:
            s = int(arg_list[5])
        if arg_list[-2] == 'f':
            uncentered = False
        else:
            uncentered = True
        if arg_list[-1] == 'f':
            noisy = False
        else:
            noisy = True

    path_to_dataset = root + 'tcl_' + '_'.join(
        [str(nps), str(ns), str(dl), str(dd), str(nl), str(s), p, a])
    if uncentered:
        path_to_dataset += '_u'
    if noisy:
        path_to_dataset += '_n'
    path_to_dataset += '.npz'

    if not os.path.exists(path_to_dataset) or s is None:
        kwargs = {"n_per_seg": nps, "n_seg": ns, "d_sources": dl, "d_data": dd, "n_layers": nl, "prior": p,
                  "activation": a, "seed": s, "batch_size": 0, "uncentered": uncentered, "noisy": noisy}
        save_data(path_to_dataset, **kwargs)
    return path_to_dataset


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='generate artificial data')
    parser.add_argument('nps', type=int, nargs='?', default=2000,
                        help='number of data points per segment')
    parser.add_argument('ns', type=int, nargs='?', default=40, help='number of segments')
    parser.add_argument('dl', type=int, nargs='?', default=2,
                        help='dimension of the latent sources')
    parser.add_argument('dd', type=int, nargs='?', default=None,
                        help='dimension of the data')
    parser.add_argument('-l', '--n-layers', type=int, default=3, dest='nl',
                        help='number of layers in generating MLP - default: 3    ')
    parser.add_argument('-s', '--seed', type=int, default=1, dest='s',
                        help='random seed of generating MLP - default: 1')
    parser.add_argument('-p', '--prior', default='gauss', dest='p',
                        help='data distribution of each independent source - default: `gauss`')
    parser.add_argument('-a', '--activation', default='xtanh', dest='a',
                        help='activation function of the generating MLP - default: `xtanh`')
    parser.add_argument('-u', '--uncentered', action='store_true', default=False,
                        help='Generate uncentered data - default False')
    parser.add_argument('-n', '--noisy', action='store_true', default=False,
                        help='Generate noisy data - default False')
    args = parser.parse_args()
    if args.dd is None:
        args.dd = 4 * args.dl

    root = 'data/'
    if not os.path.exists(root):
        os.mkdir(root)
    path_to_dataset = root + 'tcl_' + '_'.join(
        [str(args.nps), str(args.ns), str(args.dl), str(args.dd), str(args.nl), str(args.s), args.p, args.a])
    if args.uncentered:
        path_to_dataset += '_u'
    if args.noisy:
        path_to_dataset += '_noisy'
    path_to_dataset += '.npz'

    if not os.path.exists(path_to_dataset) or args.s is None:
        kwargs = {"n_per_seg": args.nps, "n_seg": args.ns, "d_sources": args.dl, "d_data": args.dd, "n_layers": args.nl,
                  "prior": args.p,
                  "activation": args.a, "seed": args.s, "batch_size": 0, "uncentered": args.uncentered,
                  "noisy": args.noisy}
        save_data(path_to_dataset, **kwargs)
