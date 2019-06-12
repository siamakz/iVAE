import argparse
import time

import torch
from torch import optim
from torch.utils.data import DataLoader

from data import SyntheticDataset, DataLoaderGPU
from metrics import mean_corr_coef as mcc
from models import iVAE

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-b', '--batch-size', type=int, default=64, help='batch size (default 64)')
    parser.add_argument('-e', '--epochs', default=5, help='number of epochs (default 5)')
    parser.add_argument('-c', '--cuda', action='store_true', default=False, help='cuda')
    parser.add_argument('-p', '--preload-gpu', action='store_true', default=False, dest='preload',
                        help='preload data on gpu')
    parser.add_argument('-l', '--lr', default=1e-3, help='learning rate (default 1e-3)')
    parser.add_argument('-s', '--seed', default=1, help='random seed (default 1)')
    args = parser.parse_args()

    torch.manual_seed(args.seed)

    # args.preload = True
    # args.cuda = True
    # args.batch_size = 128

    st = time.time()
    path = '../VAEICA/data/tcl/tcl_1000_40_2_5_3_1_gauss_xtanh.npz'
    device = torch.device('cuda' if args.cuda else 'cpu')
    print('training on {}'.format(device))

    if not args.preload:
        dset = SyntheticDataset(path, args.preload)
        loader_params = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
        train_loader = DataLoader(dset, shuffle=True, batch_size=args.batch_size, **loader_params)
    else:
        train_loader = DataLoaderGPU(path, shuffle=True, batch_size=args.batch_size)

    model = iVAE(2, 5, 40, activation='xtanh', cuda=args.cuda)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    ste = time.time()
    print('setup time: {}s'.format(ste - st))

    elbo_hist = []
    perf_hist = []

    # training loop
    for epoch in range(args.epochs):
        # elbo_epoch = torch.zeros(1).to(device)
        elbo_epoch = 0

        perf_epoch = torch.zeros(1).to(device)
        # perf_epoch = 0

        est = time.time()
        for i, (x, u, z) in enumerate(train_loader):
            model.train()
            optimizer.zero_grad()
            # transfer to GPU
            if args.cuda and not args.preload:
                x = x.cuda(async=True)
                u = u.cuda(async=True)
            # do ELBO gradient and accumulate loss
            elbo, z_est = model.elbo(x, u)
            elbo.mul(-1).backward()
            optimizer.step()

            # elbo_epoch += elbo.detach()
            elbo_epoch += elbo.item()

            perf_epoch += mcc(z, z_est.detach().to(z.device))
            # perf_epoch += mcc(z.cpu().numpy(), z_est.cpu().detach().numpy())

        # elbo_hist.append(- elbo_epoch.cpu().item() / len(train_loader))
        elbo_hist.append(- elbo_epoch / len(train_loader))

        perf_hist.append(perf_epoch.cpu().item() / len(train_loader))
        # perf_hist.append(perf_epoch / len(train_loader))

        eet = time.time()
        print('epoch {} done in: {}s'.format(epoch + 1, eet - est))
    et = time.time()
    print(elbo_hist)
    print(perf_hist)
    print('training time: {}s'.format(et - ste))
    print('total time: {}s'.format(et - st))
