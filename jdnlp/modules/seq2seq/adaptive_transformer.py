"""
Sourced from:
https://github.com/facebookresearch/adaptive-span/blob/master/models.py
"""

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

#!/usr/bin/env python3

import math

from overrides import overrides

import torch
import torch.nn as nn
import torch.nn.functional as F

from jdnlp.modules.attention.adaptive_span import AdaptiveSpan
from allennlp.modules import FeedForward, Seq2VecEncoder, Seq2SeqEncoder

import torchsnooper
import logging
logger = logging.getLogger(__name__)

# Size notations:
# B = batch_size, H = hidden_size, M = block_size, L = attn_span

def _skew(X, pad_value):
    """shift every row 1 step to right"""
    # X = B x M x L
    B, M, L = X.size()
    X = F.pad(X, (0, M + 1), value=pad_value)  # B x M x (L+M+1)
    X = X.view(B, -1)  # B x ML+MM+M
    X = X[:, :-M]  # B x ML+MM
    X = X.view(B, M, M + L)  # B x M x L+M
    return X


def _unskew(X):
    """reverse _skew operation"""
    # X = B x M x L+M
    B, M, L = X.size()
    L -= M
    X = X.view(B, -1)  # B x ML+MM
    X = F.pad(X, (0, M))  # B x ML+MM+M
    X = X.view(B, M, M + L + 1)  # B x M x L+M+1
    X = X[:, :, :L]  # B x M x L
    return X


class FeedForwardLayer(nn.Module):
    def __init__(self, hidden_size, inner_hidden_size, dropout):
        super().__init__()
        self.fc1 = nn.Linear(hidden_size, inner_hidden_size)
        self.fc2 = nn.Linear(inner_hidden_size, hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, h):
        h1 = F.relu(self.fc1(h))
        h1 = self.dropout(h1)
        h2 = self.fc2(h1)
        return h2


class SeqAttention(nn.Module):
    """Sequential self-attention layer.
    Each token will attend to its previous fixed number of steps.
    Note that attention doesn't include the current step itself.
    """
    def __init__(self, hidden_size, attn_span,
                 dropout, adapt_span_params):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.hidden_size = hidden_size # size of a single head
        self.attn_span = attn_span
        self.adapt_span_enabled = adapt_span_params['adapt_span_enabled']
        if self.adapt_span_enabled:
            self.adaptive_span = AdaptiveSpan(attn_span=attn_span, **adapt_span_params)

    def forward(self, query, key, value, key_pe):
        # query size = B x M x H
        # key, value sizes = B x (M+L) x H

        if self.adapt_span_enabled:
            # [optional] trim out memory to reduce unnecessary computation
            key, value, key_pe = self.adaptive_span.trim_memory(
                query, key, value, key_pe)

        # compute attention from context
        # B x M (dest) x (M+L) (src)
        attn_cont = torch.matmul(query, key.transpose(-1, -2))
        attn_cont = _unskew(attn_cont)  # B x M x L

        # compute the effect of position embedding
        attn_pos = torch.matmul(query, key_pe)  # B x M x L_pos
        attn = attn_cont + attn_pos

        attn = attn / math.sqrt(self.hidden_size)  # B x M X L_pos
        attn = F.softmax(attn, dim=-1)

        if self.adapt_span_enabled:
            # trim attention lengths according to the learned span
            attn = self.adaptive_span(attn)
        attn = self.dropout(attn)  # B x M X L_pos

        attn_cont = _skew(attn, 0)  # B x M X (L+M)
        out = torch.matmul(attn_cont, value)  # B x M x H

        return out

    def get_cache_size(self):
        if self.adapt_span_enabled:
            return self.adaptive_span.get_cache_size()
        else:
            return self.attn_span


class MultiHeadSeqAttention(nn.Module):
    def __init__(self, hidden_size, nb_heads, dropout, attn): # adapt_span_params):
        super().__init__()
        assert hidden_size % nb_heads == 0
        self.nb_heads = nb_heads
        self.head_dim = hidden_size // nb_heads
        
        self.attn = attn
        """
        SeqAttention(
            hidden_size=self.head_dim,
            attn_span=attn_span,
            dropout=dropout,
            adapt_span_params=adapt_span_params
        )
        """
        
        
        self.proj_query = nn.Linear(hidden_size, hidden_size, bias=False)
        self.proj_out = nn.Linear(hidden_size, hidden_size, bias=False)
        self.proj_val = nn.Linear(hidden_size, hidden_size, bias=False)
        self.proj_key = nn.Linear(hidden_size, hidden_size, bias=False)

    def head_reshape(self, x):
        K = self.nb_heads
        D = self.head_dim
        x = x.view(x.size()[:-1] + (K, D))  # B x (M+L) x K x D
        x = x.transpose(1, 2).contiguous()  # B x K x (M+L) x D
        x = x.view(-1, x.size(-2), x.size(-1))  # B_K x (M+L) x D
        return x

    def forward(self, query, key, value, key_pe):
        B = query.size(0)
        K = self.nb_heads
        D = self.head_dim
        M = query.size(1)

        query = self.proj_query(query)
        query = self.head_reshape(query)
        value = self.proj_val(value)
        value = self.head_reshape(value)
        key = self.proj_key(key)
        key = self.head_reshape(key)

        out = self.attn(query, key, value, key_pe)  # B_K x M x D
        out = out.view(B, K, M, D)  # B x K x M x D
        out = out.transpose(1, 2).contiguous()  # B x M x K x D
        out = out.view(B, M, -1)  # B x M x K_D
        out = self.proj_out(out)
        return out


# @Seq2SeqEncoder.register('transformer_seq_layer')
class TransformerSeqLayer(nn.Module):
    def __init__(self, 
                 hidden_size: int,
                 inner_hidden_size: int,
                 dropout: float,
                 attn: Seq2SeqEncoder,
        ):
        super().__init__()
        self.attn = attn # MultiHeadSeqAttention(hidden_size=hidden_size, **kargs)
        self.ff = FeedForwardLayer(hidden_size, inner_hidden_size, dropout)
        self.norm1 = nn.LayerNorm(hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size)

    def forward(self, h, h_cache, key_pe):
        # h = B x M x H
        # h_cache = B x L x H
        
        # logger.warn(h_cache)
        
        h_all = torch.cat([h_cache, h], dim=1)  # B x (M+L) x H
        attn_out = self.attn(h, h_all, h_all, key_pe)
        h = self.norm1(h + attn_out)  # B x M x H
        ff_out = self.ff(h)
        out = self.norm2(h + ff_out)  # B x M x H
        return out


@Seq2SeqEncoder.register("adaptive_transformer")
class TransformerSeq(Seq2SeqEncoder):
    def __init__(self,                
                 hidden_size: int,
                 inner_hidden_size: int, 
                 nb_heads: int, 
                 nb_layers: int,
                 attn_span: int,  
                 
                 adapt_span_params: dict,
                 dropout=0.01,
                 
                 pretrained_fp: str = None
                 ):
        super().__init__()
        
        # position embeddings
        self.key_pe = nn.Parameter(
            torch.randn(1, hidden_size // nb_heads, attn_span))

        self.layers = nn.ModuleList((
            TransformerSeqLayer(
                hidden_size=hidden_size,
                inner_hidden_size=inner_hidden_size,
                dropout=dropout, 
                attn=MultiHeadSeqAttention(
                    hidden_size=hidden_size,
                    nb_heads=nb_heads, 
                    dropout=dropout,
                    attn=SeqAttention(
                        hidden_size=hidden_size,
                        attn_span=attn_span,
                        dropout=dropout,
                        adapt_span_params={**adapt_span_params, 'nb_heads': nb_heads}
                    )
                )
            )
            for _ in range(nb_layers))
        )
        
        self.hidden_size = hidden_size
        self.h_cache = None
        
        if pretrained_fp:
            self.load_pretrained(pretrained_fp)
        
    @overrides
    def get_input_dim(self):
        return self.hidden_size
    
    @overrides
    def get_output_dim(self):
        return self.hidden_size
    
    @overrides
    def is_bidirectional(self):
        return False
    
    def load_pretrained(self, pretrained_fp): # '~/.pretrained/enwik8.pt'
        model_state_dict = torch.load(pretrained_fp)['model']
        new_model_state = {}

        # https://github.com/pytorch/pytorch/issues/3805
        for key in model_state_dict.keys():
            if 'emb' in key:
                continue
            new_model_state[key[7:]] = model_state_dict[key]
            
        logger.info(self.load_state_dict(new_model_state))
        self.requires_grad_(False)


    def init_h_cache(self, B):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        # logger.warn(f'Initializing hidden cache on device: {device}')
        self.h_cache = [torch.zeros(B, layer.attn.attn.get_cache_size(), self.hidden_size).to(device) for layer in self.layers]
    
    # @torchsnooper.snoop(watch='h_cache_next_l')
    def forward(self, h, *args, **kwargs):
        # h size = B x M x H
        
        if self.h_cache is None:
            self.init_h_cache(h.size(0))    
        
        h_cache = self.h_cache
        block_size = h.size(1)
        h_cache_next = []
        
        for l, layer in enumerate(self.layers):
            cache_size = layer.attn.attn.get_cache_size()
            if cache_size > block_size:
                h_cache_next_l = torch.cat(
                    [h_cache[l][:, -cache_size + block_size:, :], h],
                    dim=1).detach()
            else:
                h_cache_next_l = h[:, -cache_size:, :].detach()
            h_cache_next.append(h_cache_next_l)
            h = layer(h, h_cache[l], self.key_pe)  # B x M x H

        self.h_cache = h_cache_next
        return h