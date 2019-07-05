import argparse
import time

import numpy as np
import torch
from tensorboardX import SummaryWriter
from torch import optim
from torch.utils.data import DataLoader

from lib.data import SyntheticDataset, DataLoaderGPU, create_if_not_exist_dataset
from lib.metrics import mean_corr_coef as mcc
from lib.models import iVAE
from lib.utils import Logger, checkpoint

LOG_FOLDER = 'log/'
TENSORBOARD_RUN_FOLDER = 'runs/'
TORCH_CHECKPOINT_FOLDER = 'ckpt/'

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--file', default=None, help='path to data file in .npz format. (default None)')
    parser.add_argument('-x', '--data-args', type=str, default=None,
                        help='argument string to generate a dataset. '
                             'This should be of the form nps_ns_dl_dd_nl_s_p_a_u_n. '
                             'Usage explained in lib.data.create_if_not_exist_dataset. '
                             'This will overwrite the `file` argument if given. (default None). '
                             'In case of this argument and `file` argument being None, a default dataset '
                             'described in data.py will be created.')
    parser.add_argument('-b', '--batch-size', type=int, default=64, help='batch size (default 64)')
    parser.add_argument('-e', '--epochs', type=int, default=20, help='number of epochs (default 20)')
    parser.add_argument('-m', '--max-iter', type=int, default=None, help='max iters, overwrites --epochs')
    parser.add_argument('-g', '--hidden-dim', type=int, default=50, help='hidden dim of the networks (default 50)')
    parser.add_argument('-d', '--depth', type=int, default=3, help='depth (n_layers) of the networks (default 3)')
    parser.add_argument('-l', '--lr', type=float, default=1e-3, help='learning rate (default 1e-3)')
    parser.add_argument('-s', '--seed', type=int, default=1, help='random seed (default 1)')
    parser.add_argument('-c', '--cuda', action='store_true', default=False, help='train on gpu')
    parser.add_argument('-p', '--preload-gpu', action='store_true', default=False, dest='preload',
                        help='preload data on gpu for faster training.')
    parser.add_argument('-a', '--anneal', action='store_true', default=False, help='use annealing in learning')
    parser.add_argument('-n', '--no-log', action='store_true', default=False, help='run without logging')
    parser.add_argument('-q', '--log-freq', type=int, default=25, help='logging frequency (default 25).')
    args = parser.parse_args()

    print(args)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    st = time.time()

    if args.file is None:
        args.file = create_if_not_exist_dataset(root='data/', arg_str=args.data_args)
    metadata = vars(args).copy()
    del metadata['no_log'], metadata['data_args']

    device = torch.device('cuda' if args.cuda else 'cpu')
    print('training on {}'.format(torch.cuda.get_device_name(device) if args.cuda else 'cpu'))

    # load data
    if not args.preload:
        dset = SyntheticDataset(args.file, 'cpu')
        loader_params = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
        train_loader = DataLoader(dset, shuffle=True, batch_size=args.batch_size, **loader_params)
        data_dim, latent_dim, aux_dim = dset.get_dims()
        args.N = len(dset)
        metadata.update(dset.get_metadata())
    else:
        train_loader = DataLoaderGPU(args.file, shuffle=True, batch_size=args.batch_size)
        data_dim, latent_dim, aux_dim = train_loader.get_dims()
        args.N = train_loader.dataset_len
        metadata.update(train_loader.get_metadata())
    if args.max_iter is None:
        args.max_iter = len(train_loader) * args.epochs

    # define model and optimizer
    model = iVAE(latent_dim, data_dim, aux_dim, activation='lrelu', device=device, hidden_dim=args.hidden_dim,
                 anneal=args.anneal)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.1, patience=4, verbose=True)

    ste = time.time()
    print('setup time: {}s'.format(ste - st))

    # setup loggers
    logger = Logger(path=LOG_FOLDER)
    exp_id = logger.get_id()
    tensorboard_run_name = TENSORBOARD_RUN_FOLDER + 'exp' + str(exp_id) + '_'.join(
        map(str, ['', args.batch_size, args.max_iter, args.lr, args.hidden_dim, args.depth, args.anneal]))
    writer = SummaryWriter(logdir=tensorboard_run_name)
    logger.add('elbo')
    logger.add('perf')
    print('Beginning training for exp: {}'.format(exp_id))

    # training loop
    it = 0
    model.train()
    while it < args.max_iter:
        est = time.time()
        for _, (x, u, z) in enumerate(train_loader):
            it += 1
            model.anneal(args.N, args.max_iter, it)
            optimizer.zero_grad()

            if args.cuda and not args.preload:
                x = x.cuda(device=device, non_blocking=True)
                u = u.cuda(device=device, non_blocking=True)

            elbo, z_est = model.elbo(x, u)
            elbo.mul(-1).backward()
            optimizer.step()

            logger.update('elbo', -elbo.item())

            perf = mcc(z.cpu().numpy(), z_est.cpu().detach().numpy())
            logger.update('perf', perf)

            if it % args.log_freq == 0:
                logger.log()
                writer.add_scalar('data/performance', logger.get_last('perf'), it)
                writer.add_scalar('data/elbo', logger.get_last('elbo'), it)
                scheduler.step(logger.get_last('elbo'))

            if it % int(args.max_iter / 5) == 0 and not args.no_log:
                checkpoint(TORCH_CHECKPOINT_FOLDER, exp_id, it, model, optimizer,
                           logger.get_last('elbo'), logger.get_last('perf'))

        eet = time.time()
        print('epoch {} done in: {}s;\tloss: {};\tperf: {}'.format(int(it / len(train_loader)) + 1, eet - est,
                                                                   logger.get_last('elbo'), logger.get_last('perf')))

    et = time.time()
    print('training time: {}s'.format(et - ste))

    writer.close()
    if not args.no_log:
        logger.add_metadata(**metadata)
        logger.save_to_json()
        logger.save_to_npz()

    print('total time: {}s'.format(et - st))
