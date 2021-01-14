import torch


def split_shot_query(data, way, shot, query, pseudo, ep_per_batch=1):
    img_shape = data.shape[1:]
    data = data.view(ep_per_batch, way, shot + query + pseudo, *img_shape)
    x_shot, x_query, x_pseudo = data.split([shot, query, pseudo], dim=2)
    x_shot = x_shot.contiguous()
    x_pseudo = x_pseudo.contiguous()
    x_query = x_query.contiguous().view(ep_per_batch, way * query, *img_shape)
    return x_shot, x_query, x_pseudo


def make_nk_label(n, k, ep_per_batch=1):
    label = torch.arange(n).unsqueeze(1).expand(n, k).reshape(-1)
    label = label.repeat(ep_per_batch)
    return label

