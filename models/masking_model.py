import torch
import torch.nn as nn
import torch.nn.functional as F

import models
import utils
from .models import register


@register('masking-model')
class MaskingModel(nn.Module):

    def __init__(self, encoder, encoder_args, masking, masking_args,  universal, universal_args,
                 method='cos',
                 temp=10., temp_learnable=True):
        super().__init__()

        self.encoder = models.make(encoder, **encoder_args)
        self.aggregator = models.make('mean-aggregator', **{})
        masking_inplane = self.encoder.out_dim * 2
        masking_args['inplanes'] = masking_inplane
        self.masking_model = models.make(masking, **masking_args)
        in_planes = int(masking_inplane / 2)

        self.universal = models.make(universal, inplanes=in_planes, **universal_args)

        self.method = method
        if temp_learnable:
            self.temp = nn.Parameter(torch.tensor(temp))
        else:
            self.temp = temp

    def forward(self, x_shot, x_query, x_pseudo):
        shot_shape = x_shot.shape[:-3]
        query_shape = x_query.shape[:-3]
        pseudo_shape = x_pseudo.shape[:-3]
        img_shape = x_shot.shape[-3:]

        x_shot = x_shot.view(-1, *img_shape)
        x_query = x_query.view(-1, *img_shape)
        x_pseudo = x_pseudo.view(-1, *img_shape)
        x_tot = self.encoder(torch.cat([x_shot, x_query, x_pseudo], dim=0))
        x_shot, x_query, x_pseudo = x_tot[:len(x_shot)], x_tot[len(x_shot):len(x_shot) + len(x_query)], x_tot[len(
            x_shot) + len(x_query):]
        x_shot = x_shot.view(*shot_shape, *x_tot.shape[1:])
        x_query = x_query.view(*query_shape, *x_tot.shape[1:])
        x_pseudo = x_pseudo.view(*pseudo_shape, *x_tot.shape[1:])

        a_shot = self.aggregator(x_shot)
        a_pseudo = self.aggregator(x_pseudo)
        total = torch.cat((a_shot, a_pseudo), dim=-3)
        batch_shape = total.shape[:2]
        feat_shape = total.shape[2:]
        total = total.view(-1, *feat_shape)
        mask = self.masking_model(total)
        mask = mask.view(*batch_shape, *mask.shape[1:]).unsqueeze(dim=2)

        x_pseudo = torch.mul(x_pseudo, mask)
        img_shape = x_query.shape[-3:]
        x_shot = x_shot.view(-1, *img_shape)
        x_query = x_query.view(-1, *img_shape)
        x_pseudo = x_pseudo.view(-1, *img_shape)
        x_tot = self.universal(torch.cat([x_shot, x_query, x_pseudo], dim=0))
        x_shot, x_query, x_pseudo = x_tot[:len(x_shot)], x_tot[len(x_shot):len(x_shot) + len(x_query)], x_tot[len(
            x_shot) + len(x_query):]
        x_shot = x_shot.view(*shot_shape, *x_tot.shape[1:])
        x_query = x_query.view(*query_shape, *x_tot.shape[1:])
        x_pseudo = x_pseudo.view(*pseudo_shape, *x_tot.shape[1:])

        x_shot_c = torch.cat([x_shot, x_pseudo], dim=2).mean(2).view(*shot_shape[:2], -1)
        x_query = x_query.view(*query_shape, -1)
        x_shot = x_shot_c

        if self.method == 'cos':
            x_shot = F.normalize(x_shot, dim=-1)
            x_query = F.normalize(x_query, dim=-1)
            metric_ = 'dot'
        elif self.method == 'sqr':
            metric_ = 'sqr'

        logits = utils.compute_logits(
            x_query, x_shot, metric=metric_, temp=self.temp)

        return logits
