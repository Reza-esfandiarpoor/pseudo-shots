import argparse
import os
import yaml

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import pickle as pkl
from tensorboardX import SummaryWriter

import models
import utils
import utils.few_shot as fs
from data.dataset import CustomDataset
from data.sampler import EpisodicSampler
from torch.utils.data import DataLoader


def main(config):
    svname = config.get('sv_name')
    if args.tag is not None:
        svname += '_' + args.tag
    config['sv_name'] = svname
    save_path = os.path.join('./save', svname)
    utils.ensure_path(save_path)
    utils.set_log_path(save_path)
    utils.log(svname)
    writer = SummaryWriter(os.path.join(save_path, 'tensorboard'))
    yaml.dump(config, open(os.path.join(save_path, 'config.yaml'), 'w'))

    #### Dataset ####

    n_way, n_shot = config['n_way'], config['n_shot']
    n_query = config['n_query']
    n_pseudo = config['n_pseudo']
    ep_per_batch = config['ep_per_batch']

    if config.get('test_batches') is not None:
        test_batches = config['test_batches']
    else:
        test_batches = config['train_batches']

    for s in ['train', 'val', 'tval']:
        if config.get(f"{s}_dataset_args") is not None:
            config[f"{s}_dataset_args"]['data_dir'] = os.path.join(os.getcwd(), os.pardir, 'data_root')

    # train
    train_dataset = CustomDataset(config['train_dataset'], save_dir=config.get('load_encoder'),
                                  **config['train_dataset_args'])

    if config['train_dataset_args']['split'] == 'helper':
        with open(os.path.join(save_path, 'train_helper_cls.pkl'), 'wb') as f:
            pkl.dump(train_dataset.dataset_classes, f)

    train_sampler = EpisodicSampler(train_dataset, config['train_batches'], n_way, n_shot, n_query,
                                    n_pseudo, episodes_per_batch=ep_per_batch)
    train_loader = DataLoader(train_dataset, batch_sampler=train_sampler,
                                  num_workers=4, pin_memory=True)

    # tval
    if config.get('tval_dataset'):
        tval_dataset = CustomDataset(config['tval_dataset'],
                                     **config['tval_dataset_args'])

        tval_sampler = EpisodicSampler(tval_dataset, test_batches, n_way, n_shot, n_query,
                                       n_pseudo, episodes_per_batch=ep_per_batch)
        tval_loader = DataLoader(tval_dataset, batch_sampler=tval_sampler,
                                 num_workers=4, pin_memory=True)
    else:
        tval_loader = None

    # val
    val_dataset = CustomDataset(config['val_dataset'],
                                **config['val_dataset_args'])
    val_sampler = EpisodicSampler(val_dataset, test_batches, n_way, n_shot, n_query,
                                  n_pseudo, episodes_per_batch=ep_per_batch)
    val_loader = DataLoader(val_dataset, batch_sampler=val_sampler,
                            num_workers=4, pin_memory=True)


    #### Model and optimizer ####

    if config.get('load'):
        model_sv = torch.load(config['load'])
        model = models.load(model_sv)
    else:
        model = models.make(config['model'], **config['model_args'])
        if config.get('load_encoder'):
            encoder = models.load(torch.load(config['load_encoder'])).encoder
            model.encoder.load_state_dict(encoder.state_dict())
            if config.get('freeze_encoder'):
                for param in model.encoder.parameters():
                    param.requires_grad = False

    if config.get('_parallel'):
        model = nn.DataParallel(model)

    utils.log('num params: {}'.format(utils.compute_n_params(model)))

    optimizer, lr_scheduler = utils.make_optimizer(
        model.parameters(),
        config['optimizer'], **config['optimizer_args'])

    ########

    max_epoch = config['max_epoch']
    save_epoch = config.get('save_epoch')
    max_va = 0.
    timer_used = utils.Timer()
    timer_epoch = utils.Timer()

    aves_keys = ['tl', 'ta', 'tvl', 'tva', 'vl', 'va']
    trlog = dict()
    for k in aves_keys:
        trlog[k] = []

    for epoch in range(1, max_epoch + 1):
        timer_epoch.s()
        aves = {k: utils.Averager() for k in aves_keys}

        # train
        model.train()
        if config.get('freeze_bn'):
            utils.freeze_bn(model)
        writer.add_scalar('lr', optimizer.param_groups[0]['lr'], epoch)
        np.random.seed(epoch)

        for data in tqdm(train_loader, desc='train', leave=False):
            x_shot, x_query, x_pseudo = fs.split_shot_query(
                data.cuda(), n_way, n_shot, n_query, n_pseudo,
                ep_per_batch=ep_per_batch)
            label = fs.make_nk_label(n_way, n_query,
                                     ep_per_batch=ep_per_batch).cuda()

            logits = model(x_shot, x_query, x_pseudo)
            logits = logits.view(-1, n_way)
            loss = F.cross_entropy(logits, label)
            acc = utils.compute_acc(logits, label)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            aves['tl'].add(loss.item())
            aves['ta'].add(acc)

            logits = None; loss = None

            # eval
        model.eval()
        for name, loader, name_l, name_a in [
            ('tval', tval_loader, 'tvl', 'tva'),
            ('val', val_loader, 'vl', 'va')]:

            if (config.get('tval_dataset') is None) and name == 'tval':
                continue

            np.random.seed(0)
            for data in tqdm(loader, desc=name, leave=False):
                x_shot, x_query, x_pseudo = fs.split_shot_query(
                    data.cuda(), n_way, n_shot, n_query, n_pseudo,
                    ep_per_batch=ep_per_batch)
                label = fs.make_nk_label(n_way, n_query,
                                         ep_per_batch=ep_per_batch).cuda()

                with torch.no_grad():
                    logits = model(x_shot, x_query, x_pseudo)
                    logits = logits.view(-1, n_way)
                    loss = F.cross_entropy(logits, label)
                    acc = utils.compute_acc(logits, label)

                aves[name_l].add(loss.item())
                aves[name_a].add(acc)

        # post
        if lr_scheduler is not None:
            lr_scheduler.step()

        for k, v in aves.items():
            aves[k] = v.item()
            trlog[k].append(aves[k])

        t_epoch = utils.time_str(timer_epoch.t())
        t_used = utils.time_str(timer_used.t())
        t_estimate = utils.time_str(timer_used.t() / epoch * max_epoch)
        utils.log('epoch {}, train {:.4f}|{:.4f}, tval {:.4f}|{:.4f}, '
                  'val {:.4f}|{:.4f}, {} {}/{}'.format(
            epoch, aves['tl'], aves['ta'], aves['tvl'], aves['tva'],
            aves['vl'], aves['va'], t_epoch, t_used, t_estimate))

        writer.add_scalars('loss', {
            'train': aves['tl'],
            'tval': aves['tvl'],
            'val': aves['vl'],
        }, epoch)
        writer.add_scalars('acc', {
            'train': aves['ta'],
            'tval': aves['tva'],
            'val': aves['va'],
        }, epoch)

        if config.get('_parallel'):
            model_ = model.module
        else:
            model_ = model

        training = {
            'epoch': epoch,
            'optimizer': config['optimizer'],
            'optimizer_args': config['optimizer_args'],
            'optimizer_sd': optimizer.state_dict(),
        }
        save_obj = {
            'file': __file__,
            'config': config,

            'model': config['model'],
            'model_args': config['model_args'],
            'model_sd': model_.state_dict(),

            'training': training,
        }
        torch.save(save_obj, os.path.join(save_path, 'epoch-last.pth'))
        torch.save(trlog, os.path.join(save_path, 'trlog.pth'))

        if (save_epoch is not None) and epoch % save_epoch == 0:
            torch.save(save_obj,
                       os.path.join(save_path, 'epoch-{}.pth'.format(epoch)))

        if aves['va'] > max_va:
            max_va = aves['va']
            torch.save(save_obj, os.path.join(save_path, 'max-va.pth'))

        writer.flush()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--name', default=None)
    parser.add_argument('--tag', default=None)
    parser.add_argument('--helper', action='store_true')
    parser.add_argument('--embedding', required=True)
    parser.add_argument('--nshot', required=True, type=int)
    parser.add_argument('--aux-level', required=True, type=int)
    parser.add_argument('--topk', type=int)
    args = parser.parse_args()

    config = yaml.load(open(args.config, 'r'), Loader=yaml.FullLoader)

    if torch.cuda.device_count() > 1:
        config['_parallel'] = True
        str_list = [str(item) for item in range(torch.cuda.device_count())]
        config['_gpu'] = ','.join(str_list)

    config['n_shot'] = args.nshot
    flags = ''
    if args.helper:
        config['train_dataset_args']['split'] = 'helper'
        flags += '-helper'
        config[f"train_dataset_args"]['encoder_path'] = args.embedding

    for key in config.keys():
        if 'dataset_args' in key:
            config[key]['aux_level'] = args.aux_level
            if args.topk is not None:
                config[key]['top_k'] = args.topk
    if args.topk is not None:
        flags += f'-top{args.topk}'


    config['sv_name'] = f"masking-{config['train_dataset']}-{config['n_shot']}shot-aux{args.aux_level}{flags}"

    if args.embedding is not None:
        config['load_encoder'] = os.path.join(args.embedding, 'epoch-100.pth')
        config['sv_name'] += '_' + os.path.basename(args.embedding.rstrip('/'))

    if args.name is not None:
        config['sv_name'] = args.name

    main(config)

