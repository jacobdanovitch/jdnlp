"""
Full credit to:
https://github.com/fastnlp/fastNLP/blob/master/fastNLP/modules/encoder/star_transformer.py
"""


import torch
from torch import nn
import torch.nn.functional as F

import numpy as NP
from entmax import sparsemax

from typing import Dict, Optional
from overrides import overrides
from allennlp.modules import Seq2VecEncoder, Seq2SeqEncoder, TimeDistributed

import logging
logger = logging.getLogger(__name__)

@Seq2VecEncoder.register("star_transformer")
class StarTransformerPooling(Seq2VecEncoder):
    def __init__(self, hidden_size, num_layers, num_head, head_dim, dropout=0.1, max_len=None):
        super().__init__()
        self.star_transformer = StarTransformer(hidden_size, num_layers, num_head, head_dim, dropout, max_len)
    
    @overrides    
    def forward(self, x: torch.Tensor, mask: torch.Tensor=None, *args, **kwargs): 
        nodes, relay = self.star_transformer(x, mask)
        return 0.5 * (relay + nodes.max(1)[0])
    
    @overrides
    def get_input_dim(self) -> int:
        return self.star_transformer.get_input_dim()

    @overrides
    def get_output_dim(self) -> int:
        return self.star_transformer.get_output_dim()
    

@Seq2SeqEncoder.register("star_transformer") 
class StarTransformer(Seq2SeqEncoder):
    def __init__(self, hidden_size, num_layers, num_head, head_dim, dropout=0.1, max_len=None):
        super().__init__()
        self.input_dim = self.output_dim = hidden_size
        self.iters = num_layers

        self.norm = nn.ModuleList([nn.LayerNorm(hidden_size, eps=1e-6) for _ in range(self.iters)])
        self.emb_drop = nn.Dropout(dropout)
        self.ring_att = nn.ModuleList(
            [MultiHeadedRingAttention(hidden_size, nhead=num_head, head_dim=head_dim, dropout=0.0)
             for _ in range(self.iters)])
        self.star_att = nn.ModuleList(
            [MultiHeadedStarAttention(hidden_size, nhead=num_head, head_dim=head_dim, dropout=0.0)
             for _ in range(self.iters)])

        self.pos_emb = max_len and nn.Embedding(max_len, hidden_size)

    @overrides
    def forward(self, data, mask):
        """
        :param FloatTensor data: [batch, length, hidden]
        :param ByteTensor mask: [batch, length] 
        :return: [batch, length, hidden] nodes
                [batch, hidden] relay
        """
        def norm_func(f, x):
            # B, H, L, 1
            return f(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)

        B, L, H = data.size()
        mask = (mask.eq(False))  # flip the mask for masked_fill_
        smask = torch.cat([torch.zeros(B, 1, ).byte().to(mask), mask], 1)

        embs = data.permute(0, 2, 1).unsqueeze(-1)  # B H L 1
        if self.pos_emb:
            P = self.pos_emb(torch.arange(L, dtype=torch.long, device=embs.device) \
                             .view(1, L)).permute(0, 2, 1).unsqueeze(-1)  # 1 H L 1
            embs = embs + P
        embs = norm_func(self.emb_drop, embs)
        nodes = embs
        relay = embs.mean(2, keepdim=True)
        ex_mask = mask[:, None, :, None].expand(B, H, L, 1)
        r_embs = embs.view(B, H, 1, L)
        for i in range(self.iters):
            ax = torch.cat([r_embs, relay.expand(B, H, 1, L)], 2)
            nodes = F.leaky_relu(self.ring_att[i](norm_func(self.norm[i], nodes), ax=ax))
            relay = F.leaky_relu(self.star_att[i](relay, torch.cat([relay, nodes], 2), smask))
            nodes = nodes.masked_fill_(ex_mask, 0)

        nodes = nodes.view(B, H, L).permute(0, 2, 1)
        relay = relay.view(B, H)
        return nodes, relay

    @overrides
    def get_input_dim(self) -> int:
        return self.input_dim

    @overrides
    def get_output_dim(self) -> int:
        return self.output_dim
    
    @overrides
    def is_bidirectional(self) -> bool:
        return False

    
    
class MultiHeadedRingAttention(nn.Module):
    def __init__(self, nhid, nhead=10, head_dim=10, dropout=0.1):
        super(MultiHeadedRingAttention, self).__init__()
        # Multi-head Self Attention Case 1, doing self-attention for small regions
        # Due to the architecture of GPU, using hadamard production and summation are faster than dot production when unfold_size is very small
        self.WQ = nn.Conv2d(nhid, nhead * head_dim, 1)
        self.WK = nn.Conv2d(nhid, nhead * head_dim, 1)
        self.WV = nn.Conv2d(nhid, nhead * head_dim, 1)
        self.WO = nn.Conv2d(nhead * head_dim, nhid, 1)

        self.drop = nn.Dropout(dropout)
        self.nhid, self.nhead, self.head_dim, self.unfold_size = nhid, nhead, head_dim, 3

    def forward(self, x, ax=None):
        # x: B, H, L, 1, ax : B, H, X, L append features
        nhid, nhead, head_dim, unfold_size = self.nhid, self.nhead, self.head_dim, self.unfold_size
        B, H, L, _ = x.shape

        q, k, v = self.WQ(x), self.WK(x), self.WV(x)  # x: (B,H,L,1)

        if ax is not None:
            aL = ax.shape[2]
            ak = self.WK(ax).view(B, nhead, head_dim, aL, L)
            av = self.WV(ax).view(B, nhead, head_dim, aL, L)
        q = q.view(B, nhead, head_dim, 1, L)
        k = F.unfold(k.view(B, nhead * head_dim, L, 1), (unfold_size, 1), padding=(unfold_size // 2, 0)) \
            .view(B, nhead, head_dim, unfold_size, L)
        v = F.unfold(v.view(B, nhead * head_dim, L, 1), (unfold_size, 1), padding=(unfold_size // 2, 0)) \
            .view(B, nhead, head_dim, unfold_size, L)
        if ax is not None:
            k = torch.cat([k, ak], 3)
            v = torch.cat([v, av], 3)

        alphas = self.drop(F.softmax((q * k).sum(2, keepdim=True) / NP.sqrt(head_dim), 3))  # B N L 1 U
        att = (alphas * v).sum(3).view(B, nhead * head_dim, L, 1)

        ret = self.WO(att)
        return ret


class MultiHeadedStarAttention(nn.Module):
    def __init__(self, nhid, nhead=10, head_dim=10, dropout=0.1):
        # Multi-head Self Attention Case 2, a broadcastable query for a sequence key and value
        super(MultiHeadedStarAttention, self).__init__()
        self.WQ = nn.Conv2d(nhid, nhead * head_dim, 1)
        self.WK = nn.Conv2d(nhid, nhead * head_dim, 1)
        self.WV = nn.Conv2d(nhid, nhead * head_dim, 1)
        self.WO = nn.Conv2d(nhead * head_dim, nhid, 1)

        self.drop = nn.Dropout(dropout)
        self.nhid, self.nhead, self.head_dim, self.unfold_size = nhid, nhead, head_dim, 3

    def forward(self, x, y, mask=None):
        # x: B, H, 1, 1, 1 y: B H L 1
        nhid, nhead, head_dim, unfold_size = self.nhid, self.nhead, self.head_dim, self.unfold_size
        B, H, L, _ = y.shape

        q, k, v = self.WQ(x), self.WK(y), self.WV(y)

        q = q.view(B, nhead, 1, head_dim)  # B, H, 1, 1 -> B, N, 1, h
        k = k.view(B, nhead, head_dim, L)  # B, H, L, 1 -> B, N, h, L
        v = v.view(B, nhead, head_dim, L).permute(0, 1, 3, 2)  # B, H, L, 1 -> B, N, L, h
        pre_a = torch.matmul(q, k) / NP.sqrt(head_dim)
        if mask is not None:
            pre_a = pre_a.masked_fill(mask[:, None, None, :], -float('inf'))
        alphas = self.drop(F.softmax(pre_a, 3))  # B, N, 1, L
        att = torch.matmul(alphas, v).view(B, -1, 1, 1)  # B, N, 1, h -> B, N*h, 1, 1
        return self.WO(att)