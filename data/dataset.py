from PIL import Image
import numpy as np
import os
import pickle as pkl
from torch.utils.data import Dataset
from torchvision import transforms
import h5py as hp
from data.dataset_utils import prune_cls, get_helper_classes
from glob import glob


def get_imagenet_dir_tree(path, data_dir, int2wnid):
    tree_path = os.path.join(data_dir, 'imagenet_tree.pkl')
    if os.path.exists(tree_path):
        tree = pkl.load(open(tree_path, 'rb'))
        return tree
    else:
        tree = dict()
        all_classes = os.listdir(path)
        for cls in all_classes:
            tree[int2wnid.index(cls)] = glob(f"{path.rstrip('/')}/{cls}/*")

        pkl.dump(tree, open(tree_path, 'wb'))
        return tree


def get_helper_dataset(name, encoder_path, data_dir, available_aux_classes, int2wnid, imagenet_dir_tree, dataset_fp_):

    helper_classes = get_helper_classes(name, encoder_path, data_dir, available_aux_classes, int2wnid)
    num_helper_classes = len(helper_classes)
    helper_dataset_ = dict()
    for cls in helper_classes:
        helper_dataset_[cls] = imagenet_dir_tree[cls]

    for key, value in dataset_fp_.items():
        helper_dataset_[int(key)] = value

    return helper_dataset_, num_helper_classes


class CustomDataset(Dataset):

    def __init__(self, name, split, data_dir, aux_level, min_class_members=500, top_k=1, **kwargs):
        np.random.seed(0)
        augment = kwargs.get('augment')
        self.top_k = top_k
        self.aux_level = aux_level
        dataset_dir = os.path.realpath(os.path.join(data_dir, name))
        aux_dir = os.path.realpath(os.path.join(data_dir, 'imagenet'))
        self.int2wnid = pkl.load(open(os.path.join(data_dir, 'int2wnid.pkl'), 'rb'))

        self.imagenet_dir_tree = get_imagenet_dir_tree(aux_dir, data_dir, self.int2wnid)
        aux_keys = list()
        for key, value in self.imagenet_dir_tree.items():
            if len(value) >= min_class_members:
                aux_keys.append(int(key))

        all_aux_classes = set(aux_keys)
        self.similarity_matrix = np.load(os.path.join(data_dir, f'conceptnet_similarities.npy'))

        if split not in ['test', 'val']:
            test_pruned = prune_cls(name=name, split='test', aux_classes=all_aux_classes,
                                    int2wnid=self.int2wnid, aux_level=self.aux_level, data_dir=data_dir, dataset_dir=dataset_dir)
            val_pruned = prune_cls(name=name, split='val', aux_classes=all_aux_classes,
                                   int2wnid=self.int2wnid, aux_level=self.aux_level, data_dir=data_dir, dataset_dir=dataset_dir)

            self.pruned_classes = None
            all_fb = set()
            for pruned_set in [test_pruned, val_pruned]:
                for key, value in pruned_set.items():
                    all_fb = all_fb.union(set(value))
            self.available_aux_classes = all_aux_classes.difference(all_fb)

        else:
            test_pruned = prune_cls(name=name, split=split, aux_classes=all_aux_classes,
                                    int2wnid=self.int2wnid, aux_level=self.aux_level, data_dir=data_dir, dataset_dir=dataset_dir)
            self.pruned_classes = test_pruned
            self.available_aux_classes = all_aux_classes

        num_helper_classes = 0
        if split == 'helper':
            dataset_fp = hp.File(os.path.realpath(os.path.join(dataset_dir, f"{name}-train.h5")), 'r')
            dataset_fp, num_helper_classes = get_helper_dataset(name=name,
                                                                encoder_path=kwargs.get('encoder_path'),
                                                                data_dir=data_dir,
                                                                available_aux_classes=self.available_aux_classes,
                                                                int2wnid=self.int2wnid,
                                                                imagenet_dir_tree=self.imagenet_dir_tree,
                                                                dataset_fp_=dataset_fp)
        else:
            dataset_fp = hp.File(os.path.realpath(os.path.join(dataset_dir, f"{name}-{split}.h5")), 'r')

        dataset_keys = list(set(list(map(int, dataset_fp.keys()))))
        if not self.pruned_classes:
            self.pruned_classes = {item: {item} for item in dataset_keys}


        print(f"{split} dataset with top {top_k} similar classes and auxiliary data level: {aux_level}")

        self.n_classes = len(dataset_keys)
        self.dataset_classes = dataset_keys

        self.dataset = dict()
        self.dataset_class_members = dict()
        self.dataset_path = list()
        self.num_helper_images = 0

        for i, (key, value) in enumerate(dataset_fp.items()):
            num_samples = len(value)
            int_key = int(key)
            self.dataset_class_members[int_key] = np.arange(num_samples) + len(self.dataset_path)
            if i < num_helper_classes:
                self.dataset_path.extend(value)
            else:
                self.dataset_path.extend([(int_key, k) for k in range(num_samples)])
                self.dataset[int_key] = value

            if num_helper_classes == i + 1:
                self.num_helper_images = len(self.dataset_path)

        self.aux_dataset = dict()
        self.aux_dataset_class_members = dict()
        self.aux_dataset_path = list()
        for key, value in self.imagenet_dir_tree.items():
            num_samples = len(value)
            int_key = int(key)
            self.aux_dataset_class_members[int_key] = np.arange(num_samples) + len(self.aux_dataset_path)
            self.aux_dataset_path.extend(value)

        imagenet_norm_params = {'mean': [120.39586422 / 255.0, 115.59361427 / 255.0, 104.54012653 / 255.0],
                                'std': [70.68188272 / 255.0, 68.27635443 / 255.0, 72.54505529 / 255.0]}
        imagenet_normalize = transforms.Normalize(**imagenet_norm_params)

        src_normalize = imagenet_normalize
        aux_normalize = imagenet_normalize
        src_init_t = transforms.Compose([lambda x: Image.fromarray(x)])

        if name in ['mini-imagenet', 'tiered-imagenet']:
            image_size = 84
            padding = 8
        elif name == 'im800':
            image_size = 256
            padding = 16
        elif name in ['cifarfs', 'fc100']:
            image_size = 32
            padding = 4
            cifar_norm_params = {'mean': [0.5071, 0.4867, 0.4408],
                                 'std': [0.2675, 0.2565, 0.2761]}
            cifar_normalize = transforms.Normalize(**cifar_norm_params)
            src_normalize = cifar_normalize
        else:
            image_size = padding = None
            print('There is no such dataset.')
            exit()

        aux_init_t = transforms.Compose([transforms.Resize((image_size, image_size))])

        if augment:
            augment_transform = transforms.Compose([
                transforms.RandomCrop((image_size, image_size), padding=padding),
                transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
            ])
            self.src_transform = transforms.Compose([src_init_t, augment_transform, src_normalize])
            self.aux_transform = transforms.Compose([aux_init_t, augment_transform, aux_normalize])
        else:
            self.src_transform = transforms.Compose([src_init_t, transforms.ToTensor(), src_normalize])
            self.aux_transform = transforms.Compose([aux_init_t, transforms.ToTensor(), aux_normalize])

        if split == 'helper':
            self.read_src = self.read_src_helper
        else:
            self.read_src = self.read_src_no_helper

    def __len__(self):
        return self.n_classes

    def read_src_no_helper(self, index):
        entry = self.dataset_path[index]
        img = self.dataset[entry[0]][entry[1]][:]
        img = self.src_transform(img)
        return img

    def read_src_helper(self, index):
        entry = self.dataset_path[index]
        if index < self.num_helper_images:
            img = self.read_image(entry)
            img = self.aux_transform(img)
        else:
            img = self.dataset[entry[0]][entry[1]][:]
            img = self.src_transform(img)
        return img

    def read_image(self, path):
        try:
            img = Image.open(path).convert('RGB')
            return img
        except:
            dirname = os.path.dirname(path)
            wnid = os.path.basename(dirname)
            int_id = self.int2wnid.index(wnid)
            new_path = str(np.random.choice(self.imagenet_dir_tree[int_id], 1)[0])
            return self.read_image(new_path)

    def get_image(self, index_):
        if index_ < 0:
            index = -index_
            path = self.aux_dataset_path[index]
            raw_img = self.read_image(path)
            img = self.aux_transform(raw_img)
        else:
            img = self.read_src(index_)
        return img

    def __getitem__(self, index_):
        image = self.get_image(index_)
        return image


class TrainDataset(Dataset):

    def __init__(self, name, split, data_dir, aux_level, min_class_members=500, top_k=1, **kwargs):
        np.random.seed(0)
        augment = kwargs.get('augment')
        dataset_dir = os.path.realpath(os.path.join(data_dir, name))
        aux_dir = os.path.realpath(os.path.join(data_dir, 'imagenet'))

        dataset_fp = hp.File(os.path.realpath(os.path.join(dataset_dir, f"{name}-{split}.h5")), 'r')
        dataset_keys = list(set(list(map(int, dataset_fp.keys()))))
        self.int2wnid = pkl.load(open(os.path.join(data_dir, 'int2wnid.pkl'), 'rb'))
        self.similarity_matrix = np.load(os.path.join(data_dir, f'conceptnet_similarities.npy'))

        self.imagenet_dir_tree = get_imagenet_dir_tree(aux_dir, data_dir, self.int2wnid)
        aux_keys = list()
        for key, value in self.imagenet_dir_tree.items():
            if len(value) >= min_class_members:
                aux_keys.append(int(key))

        all_aux_classes = set(aux_keys)
        self.similarity_matrix = np.load(os.path.join(data_dir, f'conceptnet_similarities.npy'))
        self.separated_training_classes = {'base': dataset_keys, 'aux': list()}

        pruned_classes = list()
        if split not in ['test', 'val']:
            test_pruned = prune_cls(name=name, split='test', aux_classes=all_aux_classes,
                                    int2wnid=self.int2wnid, aux_level=aux_level, data_dir=data_dir, dataset_dir=dataset_dir)
            pruned_classes.append(test_pruned)
            val_pruned = prune_cls(name=name, split='val', aux_classes=all_aux_classes,
                                   int2wnid=self.int2wnid, aux_level=aux_level, data_dir=data_dir, dataset_dir=dataset_dir)
            pruned_classes.append(val_pruned)
            train_pruned = {item: {item} for item in dataset_keys}
            pruned_classes.append(train_pruned)
        else:
            test_pruned = prune_cls(name=name, split=split, aux_classes=all_aux_classes,
                                    int2wnid=self.int2wnid, aux_level=aux_level, data_dir=data_dir, dataset_dir=dataset_dir)
            pruned_classes.append(test_pruned)

        all_fb = set()
        for pruned_set in pruned_classes:
            for key, value in pruned_set.items():
                all_fb = all_fb.union(set(value))

        available_aux_classes = all_aux_classes.difference(all_fb)

        self.dataset = dict()
        self.dataset_class_members = dict()
        self.dataset_path = list()
        self.dataset_labels = list()
        class_counter = 0
        for key, value in dataset_fp.items():
            num_samples = value.shape[0]
            int_key = int(key)
            self.dataset_path.extend([(int_key, k) for k in range(num_samples)])
            self.dataset_labels.extend([class_counter] * num_samples)
            self.dataset[int_key] = value
            class_counter += 1

        self.num_source_images = len(self.dataset_path)

        rows = np.asarray(dataset_keys)
        cols = np.asarray(list(available_aux_classes))
        similarity_matrix = (self.similarity_matrix.ravel()[(
                cols + (rows * self.similarity_matrix.shape[1]).reshape((-1, 1))
        ).ravel()]).reshape(rows.size, cols.size)
        chosen_indices = np.argpartition(similarity_matrix, axis=1, kth=-min(similarity_matrix.shape[1], (top_k)))[:,
                         -min(similarity_matrix.shape[1], (top_k)):]

        if top_k == 0:
            chosen_indices = list()

        chosen_cls = list(set(np.asarray(list(available_aux_classes))[chosen_indices].flatten()))
        self.separated_training_classes['aux'] = chosen_cls

        for key in chosen_cls:
            int_key = int(key)
            value = self.imagenet_dir_tree[int_key]
            num_samples = len(value)
            self.dataset_path.extend(value)
            self.dataset_labels.extend([class_counter] * num_samples)
            class_counter += 1

        self.n_classes = class_counter
        imagenet_norm_params = {'mean': [120.39586422 / 255.0, 115.59361427 / 255.0, 104.54012653 / 255.0],
                                'std': [70.68188272 / 255.0, 68.27635443 / 255.0, 72.54505529 / 255.0]}
        imagenet_normalize = transforms.Normalize(**imagenet_norm_params)
        src_normalize = imagenet_normalize
        aux_normalize = imagenet_normalize
        src_init_t = transforms.Compose([lambda x: Image.fromarray(x)])

        if name in ['mini-imagenet', 'tiered-imagenet']:
            image_size = 84
            padding = 8
        elif name == 'im800':
            image_size = 256
            padding = 16
        elif name in ['cifarfs', 'fc100']:
            image_size = 32
            padding = 4
            cifar_norm_params = {'mean': [0.5071, 0.4867, 0.4408],
                                 'std': [0.2675, 0.2565, 0.2761]}
            cifar_normalize = transforms.Normalize(**cifar_norm_params)
            src_normalize = cifar_normalize
        else:
            image_size = padding = None
            print('There is no such dataset.')
            exit()

        aux_init_t = transforms.Compose([transforms.Resize((image_size, image_size))])
        if augment:
            template_transform = transforms.Compose([
                transforms.RandomCrop((image_size, image_size), padding=padding),
                transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
            ])
            self.src_transform = transforms.Compose([src_init_t, template_transform, src_normalize])
            self.aux_transform = transforms.Compose([aux_init_t, template_transform, aux_normalize])
        else:
            self.src_transform = transforms.Compose([src_init_t, transforms.ToTensor(), src_normalize])
            self.aux_transform = transforms.Compose([aux_init_t, transforms.ToTensor(), aux_normalize])

    def __len__(self):
        return len(self.dataset_labels)

    def read_image(self, path):
        try:
            img = Image.open(path).convert('RGB')
            return img
        except:
            dirname = os.path.dirname(path)
            wnid = os.path.basename(dirname)
            int_id = self.int2wnid.index(wnid)
            new_path = str(np.random.choice(self.imagenet_dir_tree[int_id], 1)[0])
            return self.read_image(new_path)

    def get_image(self, index_):
        if index_ < self.num_source_images:
            tp = self.dataset_path[index_]
            img = self.dataset[tp[0]][tp[1]][:]
            img = self.src_transform(img)
        else:
            img = self.read_image(self.dataset_path[index_])
            img = self.aux_transform(img)
        label = self.dataset_labels[index_]
        return img, label

    def __getitem__(self, index_):
        image, label = self.get_image(index_)
        return image, label
