import argparse
import os
import yaml
import pickle as pkl

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter

import models
import utils
import utils.few_shot as fs

from data.dataset import CustomDataset, TrainDataset
from data.sampler import EpisodicSampler

def main(config):
    svname = args.name
    if svname is None:
        svname = f"classifier-{config['train_dataset']}-{config['model_args']['encoder']}"
        clsfr = config['model_args']['classifier']
        if clsfr != 'linear-classifier':
            svname += '-' + clsfr

    svname += '-aux' + str(args.aux_level)

    if args.topk is not None:
        svname += f"-top{args.topk}"

    if args.tag is not None:
        svname += '_' + args.tag

    save_path = os.path.join('./save', svname)
    utils.ensure_path(save_path)
    utils.set_log_path(save_path)
    writer = SummaryWriter(os.path.join(save_path, 'tensorboard'))

    yaml.dump(config, open(os.path.join(save_path, 'config.yaml'), 'w'))

    #### Dataset ####

    for s in ['train', 'val', 'tval', 'fs', 'fs_val']:
        if config.get(f"{s}_dataset_args") is not None:
            config[f"{s}_dataset_args"]['data_dir'] = os.path.join(os.getcwd(), os.pardir, 'data_root')

    # train
    train_dataset = TrainDataset(name=config['train_dataset'], **config['train_dataset_args'])
    train_loader = DataLoader(train_dataset, config['batch_size'], shuffle=True,
                              num_workers=16, pin_memory=True, drop_last=True)

    with open(os.path.join(save_path, 'training_classes.pkl'), 'wb') as f:
        pkl.dump(train_dataset.separated_training_classes, f)

    # val
    if config.get('val_dataset'):
        eval_val = True
        val_dataset = TrainDataset(config['val_dataset'],
                                   **config['val_dataset_args'])
        val_loader = DataLoader(val_dataset, config['batch_size'],
                                num_workers=16, pin_memory=True, drop_last=True)
    else:
        eval_val = False

    # few-shot eval
    fs_loaders = {'fs_dataset': list(), 'fs_val_dataset': list()}
    for key in fs_loaders.keys():
        if config.get(key):
            ef_epoch = config.get('eval_fs_epoch')
            if ef_epoch is None:
                ef_epoch = 5
            eval_fs = True

            fs_dataset = CustomDataset(config[key],
                                       **config[key + '_args'])

            n_way = config['n_way'] if config.get('n_way') else 5
            n_query = config['n_query'] if config.get('n_query') else 15
            if config.get('n_pseudo') is not None:
                n_pseudo = config['n_pseudo']
            else:
                n_pseudo = 15
            n_batches = config['n_batches'] if config.get('n_batches') else 200
            ep_per_batch = config['ep_per_batch'] if config.get('ep_per_batch') else 4
            n_shots = [1, 5]
            for n_shot in n_shots:
                fs_sampler = EpisodicSampler(fs_dataset, n_batches, n_way, n_shot, n_query, n_pseudo, episodes_per_batch=ep_per_batch)
                fs_loader = DataLoader(fs_dataset, batch_sampler=fs_sampler,
                                       num_workers=16, pin_memory=True)
                fs_loaders[key].append(fs_loader)
        else:
            eval_fs = False

    eval_fs = False
    for key in fs_loaders.keys():
        if config.get(key):
            eval_fs = True


    #### Model and Optimizer ####

    config['model_args']['classifier_args']['n_classes'] = train_dataset.n_classes
    model = models.make(config['model'], **config['model_args'])

    if eval_fs:
        fs_model = models.make('meta-baseline', encoder=None)
        fs_model.encoder = model.encoder

    if config.get('_parallel'):
        model = nn.DataParallel(model)
        if eval_fs:
            fs_model = nn.DataParallel(fs_model)

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

    for epoch in range(1, max_epoch + 1 + 1):
        timer_epoch.s()
        aves_keys = ['tl', 'ta', 'vl', 'va']
        if eval_fs:
            for n_shot in n_shots:
                aves_keys += ['fsa-' + str(n_shot)]
                if config.get('fs_val_dataset'):
                    aves_keys += ['fsav-' + str(n_shot)]
        aves = {k: utils.Averager() for k in aves_keys}

        # train
        model.train()
        writer.add_scalar('lr', optimizer.param_groups[0]['lr'], epoch)

        for data, label in tqdm(train_loader, desc='train', leave=False):
            data, label = data.cuda(), label.cuda()
            logits = model(data)
            loss = F.cross_entropy(logits, label)
            acc = utils.compute_acc(logits, label)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            aves['tl'].add(loss.item())
            aves['ta'].add(acc)

            logits = None; loss = None

        # eval
        if eval_val:
            model.eval()
            for data, label in tqdm(val_loader, desc='val', leave=False):
                data, label = data.cuda(), label.cuda()
                with torch.no_grad():
                    logits = model(data)
                    loss = F.cross_entropy(logits, label)
                    acc = utils.compute_acc(logits, label)
                
                aves['vl'].add(loss.item())
                aves['va'].add(acc)

        if eval_fs and (epoch % ef_epoch == 0 or epoch == max_epoch + 1):
            fs_model.eval()
            for key in fs_loaders.keys():
                if len(fs_loaders[key]) == 0:
                    continue
                tag = 'v' if key == 'fs_val_dataset' else ''
                for i, n_shot in enumerate(n_shots):
                    np.random.seed(0)
                    for data in tqdm(fs_loaders[key][i],
                                        desc='fs' + tag + '-' + str(n_shot), leave=False):
                        x_shot, x_query, x_pseudo = fs.split_shot_query(
                                data.cuda(), n_way, n_shot, n_query, pseudo=n_pseudo, ep_per_batch=ep_per_batch)
                        label = fs.make_nk_label(
                                n_way, n_query, ep_per_batch=ep_per_batch).cuda()
                        with torch.no_grad():
                            logits = fs_model(x_shot, x_query, x_pseudo)
                            logits = logits.view(-1, n_way)
                            acc = utils.compute_acc(logits, label)
                        aves['fsa' + tag + '-' + str(n_shot)].add(acc)

        # post
        if lr_scheduler is not None:
            lr_scheduler.step()

        for k, v in aves.items():
            aves[k] = v.item()

        t_epoch = utils.time_str(timer_epoch.t())
        t_used = utils.time_str(timer_used.t())
        t_estimate = utils.time_str(timer_used.t() / epoch * max_epoch)

        if epoch <= max_epoch:
            epoch_str = str(epoch)
        else:
            epoch_str = 'ex'
        log_str = 'epoch {}, train {:.4f}|{:.4f}'.format(
                epoch_str, aves['tl'], aves['ta'])
        writer.add_scalars('loss', {'train': aves['tl']}, epoch)
        writer.add_scalars('acc', {'train': aves['ta']}, epoch)

        if eval_val:
            log_str += ', val {:.4f}|{:.4f}'.format(aves['vl'], aves['va'])
            writer.add_scalars('loss', {'val': aves['vl']}, epoch)
            writer.add_scalars('acc', {'val': aves['va']}, epoch)

        if eval_fs and (epoch % ef_epoch == 0 or epoch == max_epoch + 1):
            for key in fs_loaders.keys():
                if len(fs_loaders[key]) == 0:
                    continue
                tag = 'v' if key == 'fs_val_dataset' else ''
                log_str += ', fs' + tag
                for n_shot in n_shots:
                    key = 'fsa' + tag + '-' + str(n_shot)
                    log_str += ' {}: {:.4f}'.format(n_shot, aves[key])
                    writer.add_scalars('acc', {key: aves[key]}, epoch)

        if epoch <= max_epoch:
            log_str += ', {} {}/{}'.format(t_epoch, t_used, t_estimate)
        else:
            log_str += ', {}'.format(t_epoch)
        utils.log(log_str)

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
        if epoch <= max_epoch:
            torch.save(save_obj, os.path.join(save_path, 'epoch-last.pth'))

            if (save_epoch is not None) and epoch % save_epoch == 0:
                torch.save(save_obj, os.path.join(
                    save_path, 'epoch-{}.pth'.format(epoch)))

            if aves['va'] > max_va:
                max_va = aves['va']
                torch.save(save_obj, os.path.join(save_path, 'max-va.pth'))
        else:
            torch.save(save_obj, os.path.join(save_path, 'epoch-ex.pth'))

        writer.flush()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config')
    parser.add_argument('--name', default=None)
    parser.add_argument('--tag', default=None)
    parser.add_argument('--aux-level', type=int, required=True)
    parser.add_argument('--topk', type=int)
    args = parser.parse_args()

    config = yaml.load(open(args.config, 'r'), Loader=yaml.FullLoader)
    for key in config.keys():
        if 'dataset_args' in key:
            config[key]['aux_level'] = args.aux_level
            if args.topk is not None:
                config[key]['top_k'] = args.topk

    if torch.cuda.device_count() > 1:
        config['_parallel'] = True
        str_list = [str(item) for item in range(torch.cuda.device_count())]
        config['_gpu'] = ','.join(str_list)

    main(config)


