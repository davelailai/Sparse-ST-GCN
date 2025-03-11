# Copyright (c) OpenMMLab. All rights reserved.
# Modified from https://github.com/Kevinz-code/CSRA
from typing import Tuple

import torch
import torch.nn as nn
from torch.nn.modules.transformer import _get_activation_fn
from torch import Tensor
from typing import Optional
# from mmengine.model import BaseModule, ModuleList
from typing import Dict, List, Optional, Tuple

from ..builder import HEADS
from .base import BaseHead
from ...core import top_k_accuracy


@HEADS.register_module()
class MLdecoderHead(BaseHead):
    """Class-specific residual attention classifier head.

    Please refer to the `Residual Attention: ML-Decoder: Scalable and Versatile Classification Head https://arxiv.org/abs/2111.12933`_
    for details.

    Args:
        num_classes (int): Number of categories.
        in_channels (int): Number of channels in the input feature map.
        num_heads (int): Number of residual at tensor heads.
        loss (dict): Config of classification loss.
        lam (float): Lambda that combines global average and max pooling
            scores.
        init_cfg (dict, optional): The extra init config of layers.
            Defaults to use ``dict(type='Normal', layer='Linear', std=0.01)``.
    """
    def __init__(self, num_classes, num_of_groups=-1, decoder_embedding=768, 
                dropout_ratio = 0.5,
                decoder_dropout=0.1,
                alpha_adap=False,
                token_group='frame',
                initial_num_features=2048, zsl=0, **kwargs):
        super().__init__(num_classes, **kwargs)

        assert token_group in ['frame', 'temporal', 'body_part_ntu','body_part_coco','point_wise']
        self.token_group=token_group
        if token_group=='body_part_ntu':
            self.node_type = torch.tensor([0,0,0,0,1,1,1,1,2,2,2,2,3,3,3,3,4,4,4,4,0,1,1,2,2])
        elif token_group == 'body_part_coco':
            self.node_type =torch.tensor([0,0,0,0,0,1,2,1,2,1,2,3,4,3,4,3,4])
        
        self.dropout_ratio=dropout_ratio
        if self.dropout_ratio != 0:
            self.dropout = nn.Dropout(p=self.dropout_ratio)

        embed_len_decoder = 100 if num_of_groups < 0 else num_of_groups
        if embed_len_decoder > num_classes:
            embed_len_decoder = num_classes

         # switching to 768 initial embeddings
        decoder_embedding = 768 if decoder_embedding < 0 else decoder_embedding
        embed_standart = nn.Linear(initial_num_features, decoder_embedding)
        self.alpha_adap=alpha_adap
        if self.alpha_adap:
            self.alpha = nn.Parameter(torch.ones(1))

        if not zsl:
            query_embed = nn.Embedding(embed_len_decoder, decoder_embedding)
            # query_embed.requires_grad_(False)
        else:
            query_embed = None

        # self.DagmaMLP= DagmaMLP(dims=[decoder_embedding, 10, 1], bias=True, dtype=query_embed.type())
        
     
        # decoder
        decoder_dropout = decoder_dropout
        num_layers_decoder = 1
        dim_feedforward = 2048
        layer_decode = TransformerDecoderLayerOptimal(d_model=decoder_embedding,
                                                      dim_feedforward=dim_feedforward, dropout=decoder_dropout)
        self.decoder = nn.TransformerDecoder(layer_decode, num_layers=num_layers_decoder)
        self.decoder.embed_standart = embed_standart
        self.decoder.query_embed = query_embed
        self.zsl = zsl

        if self.zsl:
            if decoder_embedding != 300:
                self.wordvec_proj = nn.Linear(300, decoder_embedding)
            else:
                self.wordvec_proj = nn.Identity()
            self.decoder.duplicate_pooling = torch.nn.Parameter(torch.Tensor(decoder_embedding, 1))
            self.decoder.duplicate_pooling_bias = torch.nn.Parameter(torch.Tensor(1))
            self.decoder.duplicate_factor = 1
        else:
            # group fully-connected
            self.decoder.num_classes = num_classes
            self.decoder.duplicate_factor = int(num_classes / embed_len_decoder + 0.999)
            self.decoder.duplicate_pooling = torch.nn.Parameter(
                torch.Tensor(embed_len_decoder, decoder_embedding, self.decoder.duplicate_factor))
            self.decoder.duplicate_pooling_bias = torch.nn.Parameter(torch.Tensor(num_classes))
        torch.nn.init.xavier_normal_(self.decoder.duplicate_pooling)
        torch.nn.init.constant_(self.decoder.duplicate_pooling_bias, 0)
        self.decoder.group_fc = GroupFC(embed_len_decoder)
        self.train_wordvecs = None
        self.test_wordvecs = None
        self.logsoftmax = nn.LogSoftmax(dim=-1)
    def init_weights(self):
        pass
    
    def skleton_feature(self,x):
        pool = nn.AdaptiveAvgPool2d(1)
        N, M, C, T, V = x.shape
        x = x.reshape(N * M, C, T, V)
        if self.token_group=='frame':
            x=x.mean(-1)
        if self.token_group=='temporal':
            x=x.mean(-2)
        if self.token_group=='body_part_ntu' or self.token_group=='body_part_coco':
            # Compute the sum of x along the last dimension based on the index
            index = self.node_type.to(x.device)
            # Compute the group average on the last dimension
            group_means = []
            for i in range(index.max() + 1):
                # Select elements of x based on the index
                group_x = torch.index_select(x, dim=-1, index=(index == i).nonzero(as_tuple=False).squeeze())
                # Calculate the mean along the last dimension
                group_mean = torch.mean(group_x, dim=-1, keepdim=True)
                group_means.append(group_mean)
            # Stack the group means along the last dimension
            group_means = torch.stack(group_means, dim=-1)
            x = group_means.squeeze(-2)
        if self.token_group=='point_wise':
            x = x.flatten(2)
        # 
        # x = pool(x)
        x = x.reshape(N, M, C,-1)
        x = x.mean(dim=1)
        return x
    
    def forward(self, x):
        x = self.skleton_feature(x)
        # if self.dropout is not None:
        #     x  = self.dropout(x)
        
        # x = self.pre_logits(x)
        if len(x.shape) == 4:  # [bs,2048, 7,7]
            embedding_spatial = x.flatten(2).transpose(1, 2)
        else:  # [bs, 197,468]
            embedding_spatial = x.transpose(1, 2)

        embedding_spatial_786 = self.decoder.embed_standart(embedding_spatial)
        embedding_spatial_786 = torch.nn.functional.relu(embedding_spatial_786, inplace=True)

        bs = embedding_spatial_786.shape[0]
        if self.zsl:
            query_embed = torch.nn.functional.relu(self.wordvec_proj(self.decoder.query_embed))
        else:
            query_embed = self.decoder.query_embed.weight
        # tgt = query_embed.unsqueeze(1).repeat(1, bs, 1)
        tgt = query_embed.unsqueeze(1).expand(-1, bs, -1)  # no allocation of memory with expand
        h = self.decoder(tgt, embedding_spatial_786.transpose(0, 1))  # [embed_len_decoder, batch, 768]
        h = h.transpose(0, 1)

        if self.dropout_ratio != 0:
            h  = self.dropout(h)

        out_extrap = torch.zeros(h.shape[0], h.shape[1], self.decoder.duplicate_factor, device=h.device, dtype=h.dtype)
        self.decoder.group_fc(h, self.decoder.duplicate_pooling, out_extrap)
        if not self.zsl:
            h_out = out_extrap.flatten(1)[:, :self.decoder.num_classes]
        else:
            h_out = out_extrap.flatten(1)
        h_out += self.decoder.duplicate_pooling_bias
        
        
        logits = h_out

        return logits
    
    def loss(self, cls_score, label, **kwargs):
        """Calculate the loss given output ``cls_score``, target ``label``.
        Args:
            cls_score (torch.Tensor): The output of the model.
            label (torch.Tensor): The target output of the model.
        Returns:
            dict: A dict containing field 'loss_cls'(mandatory)
            and 'top1_acc', 'top5_acc'(optional).
        """
        losses = dict()
        if label.shape == torch.Size([]):
            label = label.unsqueeze(0)
        elif label.dim() == 1 and label.size()[0] == self.num_classes \
                and cls_score.size()[0] == 1:
            label = label.unsqueeze(0)

        if not self.multi_class and cls_score.size() != label.size():
            top_k_acc = top_k_accuracy(cls_score.detach().cpu().numpy(),
                                       label.detach().cpu().numpy(), (1, 5))
            losses['top1_acc'] = torch.tensor(
                top_k_acc[0], device=cls_score.device)
            losses['top5_acc'] = torch.tensor(
                top_k_acc[1], device=cls_score.device)

        elif self.multi_class and self.label_smooth_eps != 0:
            label = ((1 - self.label_smooth_eps) * label + self.label_smooth_eps / self.num_classes)

        loss_cls = self.loss_cls(cls_score, label, **kwargs)
        if self.alpha_adap:
            loss_cls=loss_cls*self.alpha
        # loss_cls may be dictionary or single tensor
        if isinstance(loss_cls, dict):
            losses.update(loss_cls)
        else:
            losses['loss_cls'] = loss_cls

        return losses
@torch.jit.script
class GroupFC(object):
    def __init__(self, embed_len_decoder: int):
        self.embed_len_decoder = embed_len_decoder

    def __call__(self, h: torch.Tensor, duplicate_pooling: torch.Tensor, out_extrap: torch.Tensor):
        for i in range(h.shape[1]):
            h_i = h[:, i, :]
            if len(duplicate_pooling.shape)==3:
                w_i = duplicate_pooling[i, :, :]
            else:
                w_i = duplicate_pooling
            out_extrap[:, i, :] = torch.matmul(h_i, w_i)


class TransformerDecoderLayerOptimal(nn.Module):
    def __init__(self, d_model, nhead=8, dim_feedforward=2048, dropout=0.1, activation="relu",
                 layer_norm_eps=1e-5) -> None:
        super(TransformerDecoderLayerOptimal, self).__init__()
        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.dropout = nn.Dropout(dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)

        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.norm3 = nn.LayerNorm(d_model, eps=layer_norm_eps)

        self.activation = _get_activation_fn(activation)

    def __setstate__(self, state):
        if 'activation' not in state:
            state['activation'] = torch.nn.functional.relu
        super(TransformerDecoderLayerOptimal, self).__setstate__(state)

    def forward(self, tgt: Tensor, memory: Tensor, tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None) -> Tensor:
        tgt = tgt + self.dropout1(tgt)
        tgt = self.norm1(tgt)
        tgt2 = self.multihead_attn(tgt, memory, memory)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt
