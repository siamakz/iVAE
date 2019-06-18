import json
import os

import numpy as np
import torch


def make_dir(dir_name):
    if dir_name[-1] != '/':
        dir_name += '/'
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
    return dir_name


def make_file(file_name):
    if not os.path.exists(file_name):
        open(file_name, 'a').close()
    return file_name


def get_exp_id(log_folder):
    log_folder = make_dir(log_folder)
    helper_id_file = log_folder + '.expid'
    helper_id_file = make_file(helper_id_file)
    with open(helper_id_file, 'r+') as file:
        eid = 1
        for _ in file:
            eid += 1
        file.write(str(eid) + '\n')
    return eid


def from_log(args, argv, logpath):
    """
    read from log, and allow change of arguments
    assumes that arguments are assigned using an = sign
    assumes that the first argument is --from-log. so argv[1] is of the form --from-log=id
    everything that comes after --from-log in sys.argv will be resolved and its value substituted for the one in the log
    """
    i = args.from_log
    d = {}
    new_d = vars(args).copy()
    args_not_from_log = []
    add_to_log = False
    if len(argv) > 2:
        add_to_log = True
    for a in argv[1:]:  # start from 2 if the from-log value is to be overwritten by the one in the log
        sp = a.split('=')
        args_not_from_log.append(sp[0][2:].replace('-', '_'))
    file = open(logpath)
    for line in file:
        d = json.loads(line)
        if d['id'] == i:
            break
    file.close()
    for a in args_not_from_log:
        d.pop(a)
    del d['id'], d['train_perf'], d['test_perf']
    new_d.update(d)
    return new_d, add_to_log


def checkpoint(path, exp_id, iteration, model, optimizer, loss, perf):
    sub_path = make_dir(path + str(exp_id) + '/')
    weights_path = sub_path + str(exp_id) + '_ckpt_' + str(iteration) + '.pth'
    print('.. checkpoint at iteration {} ..'.format(iteration))
    torch.save({'iteration': iteration,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss,
                'perf': perf},
               weights_path)


class RunningAverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, momentum=0.97):
        self.momentum = momentum
        self.val = None
        self.avg = 0

    def reset(self):
        self.val = None
        self.avg = 0

    def update(self, val):
        if self.val is None:
            self.avg = val
        else:
            self.avg = self.avg * self.momentum + val * (1 - self.momentum)
        self.val = val


class Averager:
    def __init__(self):
        self.val = 0
        self.count = 0
        self.avg = 0
        self.sum = 0

    def reset(self):
        self.val = 0
        self.count = 0
        self.avg = 0
        self.sum = 0

    def update(self, val):
        self.val = val
        self.sum += val
        self.count += 1
        self.avg = self.sum / self.count


class Logger:
    def __init__(self, exp_id, path='log/', **metadata):
        self.path = path
        self.reset()
        self.metadata = metadata
        self.exp_id = exp_id
        self.log_dict = {}
        self.running_means = {}

    def add(self, key):
        self.running_means.update({key: Averager()})
        self.log_dict.update({key: []})

    def update(self, key, val):
        self.running_means[key].update(val)

    def reset_means(self):
        for key in self.keys():
            self.running_means[key].reset()

    def reset(self):
        self.log_dict = {}
        self.running_means = {}

    def log(self):
        for key in self.keys():
            self.log_dict[key].append(self.running_means[key].avg)
        self.reset_means()

    def get_last(self, key):
        return self.log_dict[key][-1]

    def save_to_npz(self):
        data_path = make_dir(self.path + 'data/')
        path = data_path + str(self.exp_id) + '.npz'
        for k, v in self.log_dict.items():
            self.log_dict[k] = np.array(v)
        np.savez_compressed(path, **self.log_dict)

    def save_to_json(self):
        path = make_file(self.path + 'log.json')
        with open(path, 'a') as file:
            log = {'id': self.exp_id}
            for k in self.keys():
                log.update({k: self.get_last(k)})
            log.update({'metadata': self.metadata})
            json.dump(log, file)
            file.write('\n')

    def add_metadata(self, **metadata):
        self.metadata.update(metadata)

    def __len__(self):
        return len(self.log_dict)

    def keys(self):
        return self.running_means.keys()
