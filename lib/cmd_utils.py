import os
import sys
import argparse

from lib.data import create_if_not_exist_dataset

def parse_main_args(line):
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--file', default=None, help='path to data file')
    parser.add_argument('-x', '--data-args', type=str, default=None, help='string to generate new dataset')
    parser.add_argument('-b', '--batch-size', type=int, default=64, help='batch size (default 64)')
    parser.add_argument('-e', '--epochs', type=int, default=5, help='number of epochs (default 5)')
    parser.add_argument('-m', '--max-iter', type=int, default=None, help='max iters, overwrites --epochs')
    parser.add_argument('-g', '--hidden-dim', type=int, default=50, help='hidden dim of the networks (default 50)')
    parser.add_argument('-d', '--depth', type=int, default=3, help='depth (n_layers) of the networks (default 3)')
    parser.add_argument('-l', '--lr', type=float, default=1e-3, help='learning rate (default 1e-3)')
    parser.add_argument('-s', '--seed', type=int, default=1, help='random seed (default 1)')
    parser.add_argument('-c', '--cuda', action='store_true', default=False, help='cuda')
    parser.add_argument('-p', '--preload-gpu', action='store_true', default=False, dest='preload',
                        help='preload data on gpu')
    parser.add_argument('-a', '--anneal', action='store_true', default=False, help='use annealing in learning')
    parser.add_argument('-n', '--no-log', action='store_true', default=False, help='run without logging')
    parser.add_argument('-q', '--log-freq', type=int, default=25, help='logging frequency')
    args = parser.parse_args(line)
    return args

def seedify(arg_file):
    srange = [1, 1]
    args = sys.argv
    if len(args) == 1:
        pass
    elif len(args) == 2:
        srange = [1, int(args[1])]
    elif len(args) == 3:
        srange = [int(args[1]), int(args[2])]
    else:
        raise Exception('wrong usage')
    with open(arg_file, 'r') as f:
        seeded_name = arg_file.split('.txt')[0]+'_seeded.txt'
        with open(seeded_name, 'w') as sf:
            for line in f:
                for s in range(srange[0], srange[1]+1):
                    sf.write(line.split('\n')[0] + ' -s' + str(s) +'\n')

def create_dataset_before(args_file):
    with open(args_file, 'r') as f:
        for line in f:
            args = parse_main_args(line.split())
            create_if_not_exist_dataset(root='data/', arg_str=args.data_args)

def assign_cluster(args_file):
    with open(args_file, 'r') as f:
        fcpu = open(args_file.split('.txt')[0]+'_cpu.txt', 'w')
        fgpu = open(args_file.split('.txt')[0]+'_gpu.txt', 'w')
        cc = 0
        cg = 0
        for line in f:
            if '-c' in line.split() or '-cp' in line.split():
                fgpu.write(line)
                cg += 1
            else:
                fcpu.write(line)
                cc +=1
        fcpu.close()
        fgpu.close()
        print('Total args to be run on gpu: {}'.format(cg))
        print('Total args to be run on cpu: {}'.format(cc))

