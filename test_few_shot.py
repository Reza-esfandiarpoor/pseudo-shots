import argparse
import yaml

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import scipy.stats
from tqdm import tqdm
import os

import models
import utils
import utils.few_shot as fs

from data.dataset import CustomDataset
from data.sampler import EpisodicSampler
from torch.utils.data import DataLoader


def mean_confidence_interval(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    se = scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n - 1)
    return h


def main(config):

    config['dataset_args']['data_dir'] = os.path.join(os.getcwd(), os.pardir, 'data_root')
    dataset = CustomDataset(name=config['dataset'], **config['dataset_args'])
    n_way = 5
    n_shot = config['n_shot']
    n_query = config.get('n_query') if config.get('n_query') is not None else 15
    n_pseudo = config['n_pseudo'] if config.get('n_pseudo') is not None else 15
    n_batch = config['train_batches'] if config.get('train_batches') is not None else 200
    ep_per_batch = config['ep_per_batch'] if config.get('ep_per_batch') is not None else 4

    batch_sampler = EpisodicSampler(dataset, n_batch, n_way, n_shot, n_query, n_pseudo, episodes_per_batch=ep_per_batch)
    loader = DataLoader(dataset, batch_sampler=batch_sampler,
                            num_workers=4, pin_memory=True)

    model_sv = torch.load(config['load'])
    model = models.load(model_sv)
    if config.get('fs_dataset'):
        fs_model = models.make('meta-baseline', encoder=None)
        fs_model.encoder = model.encoder
        model = fs_model

    if config.get('_parallel'):
        model = nn.DataParallel(model)

    model.eval()

    # testing
    aves_keys = ['vl', 'va']
    aves = {k: utils.Averager() for k in aves_keys}

    test_epochs = args.test_epochs
    np.random.seed(0)
    va_lst = []
    for epoch in range(1, test_epochs + 1):
        for data in tqdm(loader, desc=f"eval: {epoch}", leave=False):
            x_shot, x_query, x_pseudo = fs.split_shot_query(
                    data.cuda(), n_way, n_shot, n_query, n_pseudo,
                    ep_per_batch=ep_per_batch)

            with torch.no_grad():
                logits = model(x_shot, x_query, x_pseudo)
                logits = logits.view(-1, n_way)
                label = fs.make_nk_label(n_way, n_query,
                        ep_per_batch=ep_per_batch).cuda()

                loss = F.cross_entropy(logits, label)
                acc = utils.compute_acc(logits, label)

                aves['vl'].add(loss.item(), len(data))
                aves['va'].add(acc, len(data))
                va_lst.append(acc)

        utils.log('test epoch {}: acc={:.2f} +- {:.2f} (%), loss={:.4f}'.format(
                epoch, aves['va'].item() * 100,
                mean_confidence_interval(va_lst) * 100,
                aves['vl'].item()), filename='test_log.txt')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', required=True)
    parser.add_argument('--test-epochs', type=int, default=2)
    parser.add_argument('--nshot', type=int)
    args = parser.parse_args()

    config = yaml.load(open(os.path.join(args.dir, 'config.yaml'), 'r'), Loader=yaml.FullLoader)
    shot_tag = ''
    if args.nshot is not None:
        config['n_shot'] = args.nshot
        shot_tag = f"{args.nshot}shot"

    if torch.cuda.device_count() > 1:
        config['_parallel'] = True
        str_list = [str(item) for item in range(torch.cuda.device_count())]
        config['_gpu'] = ','.join(str_list)

    if config.get('sv_name') is None:
        config['sv_name'] = os.path.basename(args.dir.rstrip('/'))

    if os.path.basename(args.dir.rstrip('/')).startswith('masking'):
        config['load'] = os.path.join(args.dir, 'max-va.pth')
    elif os.path.basename(args.dir.rstrip('/')).startswith('classifier'):
        config['load'] = os.path.join(args.dir, 'epoch-100.pth')
    else:
        raise ValueError('Unknown model.')



    if config.get('fs_dataset') is not None:
        config['dataset'] = config['fs_dataset']
        config['dataset_args'] = config['fs_dataset_args']
    elif config.get('tval_dataset') is not None:
        config['dataset'] = config['tval_dataset']
        config['dataset_args'] = config['tval_dataset_args']

    for entry in ['load_encoder', 'freeze_encoder']:
        if config.get(entry):
            del config[entry]

    utils.set_log_name(f"test-{shot_tag}.txt")

    main(config)

