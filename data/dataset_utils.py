import numpy as np
import os
import pickle as pkl
import h5py as hp
from pathlib import Path
import nltk
from nltk.corpus import wordnet as wn
from anytree import Node


def get_pruning_data(name, dataset_dir, **kwargs):
    path = os.path.join(dataset_dir, f'{name}-pruning.pkl')
    if os.path.exists(path):
        data = pkl.load(open(path, 'rb'))
        return data
    else:
        data = create_pruning_data(name, kwargs.get('data_dir'))
        pkl.dump(data, open(path, 'wb'))
        return data


def prune_cls(name, split, dataset_dir, aux_classes, aux_level, int2wnid, **kwargs):
    pruned = dict()
    pruning_data = get_pruning_data(name, dataset_dir, **kwargs)
    pruning_data = pruning_data[split][f"l{aux_level}"]
    for key, value in pruning_data.items():
        classes = [int2wnid.index(x) for x in value if x in int2wnid]
        pruned[key] = list(aux_classes.intersection(classes))
    return pruned


def get_helper_classes(name, encoder_path, data_dir, available_aux_classes, int2wnid):
    training_classes = pkl.load(open(os.path.join(encoder_path, 'training_classes.pkl'), 'rb'))
    aux_int_id = training_classes['aux']
    base_org = training_classes['base']


    if name in ['cifarfs', 'fc100']:
        cifarwnids = pkl.load(open(Path(data_dir).joinpath('cifar_wnids.pkl'), 'rb'))
        base_int_id = list()
        for cls in base_org:
            intersection_classes = list(set(int2wnid).intersection(cifarwnids[int(cls)]))
            base_int_id.extend([int2wnid.index(item) for item in intersection_classes])
    else:
        base_int_id = base_org

    all_training_classes = base_int_id + aux_int_id

    available_aux_for_helepr = list(available_aux_classes.difference(all_training_classes))

    sims = np.load(Path(data_dir).joinpath('conceptnet_similarities.npy').as_posix())
    sims = sims[available_aux_for_helepr, :]
    sims = sims[:, all_training_classes]
    least_similar_indices = list(np.argsort(np.sum(sims, axis=1)))[:len(base_org)]
    helper_classes = [available_aux_for_helepr[i] for i in least_similar_indices]

    return helper_classes


def create_pruning_data(name, data_dir):
    root = Path(data_dir)
    if name in ['cifarfs', 'fc100']:
        with open(root.joinpath('cifar_wnids.pkl'), 'rb') as f:
            cifar_wnids = pkl.load(f)
    int2wnid = pkl.load(open(root.joinpath('int2wnid.pkl'), 'rb'))

    stat_dict = dict()
    for split in ['train', 'test', 'val']:
        stat_dict[split] = dict()
        with hp.File(root.joinpath(f"{name}/{name}-{split}.h5"), 'r') as f:
            cls_ids = list(map(int, list(f.keys())))
        src_classes = list()
        if name in ['cifarfs', 'fc100']:
            for cls_id in cls_ids:
                src_classes.extend(cifar_wnids[cls_id])
        else:
            src_classes = [int2wnid[cls_id] for cls_id in cls_ids]
        for i in range(4):
            stat_dict[split][f"l{i}"] = {cls_id: list() for cls_id in cls_ids}
            output = prune(src_classes, i, data_dir)
            if name in ['cifarfs', 'fc100']:
                for cls_id in cls_ids:
                    for wnid in cifar_wnids[cls_id]:
                        stat_dict[split][f"l{i}"][cls_id].extend(output[wnid])
            else:
                for cls_id in cls_ids:
                    wnid = int2wnid[cls_id]
                    stat_dict[split][f"l{i}"][cls_id].extend(output[wnid])

    return stat_dict


def prune(src_classes_, aux_level, data_dir):
    wordnet_tree = get_wordnet_tree(data_dir)
    node_dict = wordnet_tree['node_dict']
    pruned = dict()
    for cls in src_classes_:
        pruned[cls] = list()
        curr_nodes = node_dict[cls]
        for cn in curr_nodes:
            path = list(cn.iter_path_reverse())
            for i in range(min(len(path), aux_level + 1)):
                anc = path[i]
                desc = anc.descendants
                pruned[cls].extend([item.name for item in (list(desc) + [anc])])

    return pruned


def get_wordnet_tree(data_dir):
    path = os.path.join(data_dir, 'wordnet_tree.pkl')
    if os.path.exists(path):
        tree = pkl.load(open(path, 'rb'))
        return tree
    else:
        nltk.download('wordnet')
        all_synsets = list(wn.all_synsets('n'))
        node_dict = {"n{:08d}".format(id_s.offset()): list() for id_s in all_synsets}

        def build(nodes):
            if isinstance(nodes, list):
                children_ = [build(subtree) for subtree in nodes[1:]]
                node_name = "n{:08d}".format(nodes[0].offset())
                node_ = Node(name=node_name, children=children_)
                node_dict[node_name].append(node_)
            else:
                node_name = "n{:08d}".format(nodes.offset())
                node_ = Node(name=node_name)
                node_dict[node_name].append(node_)
            return node_

        wntree_ = wn.synset('entity.n.01').tree(lambda s: s.hyponyms())
        root_node = build(wntree_)
        wordnet_tree = {
            'root_node': root_node,
            'node_dict': node_dict
        }
        input('do you wanna overwrite the tree file?')
        exit()
        pkl.dump(wordnet_tree, open(path, 'wb'))
        return wordnet_tree

