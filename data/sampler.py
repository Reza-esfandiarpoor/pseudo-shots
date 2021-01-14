import numpy as np
import torch

class EpisodicSampler:

    def __init__(self, dataset, n_batch, n_way, n_shot, n_query, n_pseudo, episodes_per_batch=1):
        self.dataset = dataset
        self.n_batch = n_batch
        self.n_way = n_way
        self.n_shot = n_shot
        self.n_query = n_query
        self.n_pseudo = n_pseudo
        self.episodes_per_batch = episodes_per_batch
        self.top_k = self.dataset.top_k

        self.class_members = self.dataset.dataset_class_members
        self.aux_class_members = self.dataset.aux_dataset_class_members
        self.all_class_ids = self.dataset.dataset_classes
        self.available_aux_classes = self.dataset.available_aux_classes
        self.pruned_classes = self.dataset.pruned_classes

    def __len__(self):
        return self.n_batch

    def __iter__(self):
        for i_batch in range(self.n_batch):
            batch = []
            for i_ep in range(self.episodes_per_batch):
                episode = []
                classes = np.random.choice(self.all_class_ids, self.n_way, replace=False)
                pruned = set().union(*[self.pruned_classes[i] for i in classes])
                pruned = pruned.union(classes)
                available_aux_classes = self.available_aux_classes.difference(pruned)
                available_aux_classes = np.asarray(list(available_aux_classes))

                rows = classes
                cols = available_aux_classes
                sim_matrix = (self.dataset.similarity_matrix.ravel()[(
                        cols + (rows * self.dataset.similarity_matrix.shape[1]).reshape((-1, 1))
                ).ravel()]).reshape(rows.size, cols.size)

                chosen_indices = np.argpartition(sim_matrix, axis=1, kth=-min(self.top_k, sim_matrix.shape[1]))[:, -min(self.top_k, sim_matrix.shape[1]):]
                chosen_cls = np.asarray(available_aux_classes)[chosen_indices]
                for c, p in zip(classes, chosen_cls):
                    support_samples = np.random.choice(self.class_members[c], self.n_shot + self.n_query, replace=False)
                    top_sim_aux_samples = list()
                    for aux_c in p:
                        top_sim_aux_samples.extend(self.aux_class_members[aux_c])
                    pseudo_samples = -np.random.choice(top_sim_aux_samples, self.n_pseudo, replace=False)
                    all_ = np.concatenate((support_samples, pseudo_samples), axis=0)
                    episode.append(torch.from_numpy(all_))
                episode = torch.stack(episode, dim=0)
                batch.append(episode)
            batch = torch.stack(batch, dim=0)
            yield batch.view(-1)
