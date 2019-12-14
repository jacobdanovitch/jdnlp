import torch
from torch import nn
import torch.nn.functional as F

from entmax import sparsemax

import random, math

from typing import Dict, Optional
from overrides import overrides
from allennlp.modules import Seq2VecEncoder, Seq2SeqEncoder

import logging
logger = logging.getLogger(__name__)


class SelfAttentionNarrow(nn.Module):

    def __init__(self, emb, heads=8, mask=False):
        """
        :param emb:
        :param heads:
        :param mask:
        """

        super().__init__()

        assert emb % heads == 0, f'Embedding dimension ({emb}) should be divisible by nr. of heads ({heads})'

        self.emb = emb
        self.heads = heads
        self.mask = mask

        s = emb // heads
        # - We will break the embedding into `heads` chunks and feed each to a different attention head

        self.tokeys    = nn.Linear(s, s, bias=False)
        self.toqueries = nn.Linear(s, s, bias=False)
        self.tovalues  = nn.Linear(s, s, bias=False)

        self.unifyheads = nn.Linear(heads * s, emb)

    def forward(self, x):
        b, t, e = x.size()
        h = self.heads

        keys    = self.tokeys(x).view(b, t, h, e)
        queries = self.toqueries(x).view(b, t, h, e)
        values  = self.tovalues(x).view(b, t, h, e)

        dot = torch.einsum('bthe,bihe->bhti', queries, keys) / math.sqrt(e)
        # dot = sparsemax(dot, dim=-1)
        dot = F.softmax(dot, dim=-1)

        out = torch.einsum('bhtd,bdhe->bthe', dot, values)

        # we can move reshape of weights to init; I left it here just to compare with the original implementation
        out = torch.einsum('bthe,khe->btk', out, self.unifyheads.weight.view(e,h,e)) 
        return out + self.unifyheads.bias
    
    def _forward(self, x):
        b, t, e = x.size()
        h = self.heads
        assert e == self.emb, f'Input embedding dim ({e}) should match layer embedding dim ({self.emb})'

        s = e // h
        x = x.view(b, t, h, s)

        keys    = self.tokeys(x)
        queries = self.toqueries(x)
        values  = self.tovalues(x)

        assert keys.size() == (b, t, h, s)
        assert queries.size() == (b, t, h, s)
        assert values.size() == (b, t, h, s)

        # Compute scaled dot-product self-attention

        # - fold heads into the batch dimension
        keys = keys.transpose(1, 2).contiguous().view(b * h, t, s)
        queries = queries.transpose(1, 2).contiguous().view(b * h, t, s)
        values = values.transpose(1, 2).contiguous().view(b * h, t, s)

        queries = queries / (e ** (1/4))
        keys    = keys / (e ** (1/4))
        # - Instead of dividing the dot products by sqrt(e), we scale the keys and values.
        #   This should be more memory efficient

        # - get dot product of queries and keys, and scale
        dot = torch.bmm(queries, keys.transpose(1, 2))

        assert dot.size() == (b*h, t, t)

        if self.mask: # mask out the upper half of the dot matrix, excluding the diagonal
            mask_(dot, maskval=float('-inf'), mask_diagonal=False)

        dot = F.softmax(dot, dim=-1)
        # dot = sparsemax(dot, dim=-1)
        # - dot now has row-wise self-attention probabilities

        # apply the self attention to the values
        out = torch.bmm(dot, values).view(b, h, t, s)

        # swap h, t back, unify heads
        out = out.transpose(1, 2).contiguous().view(b, t, s * h)

        return self.unifyheads(out)
    
    
@Seq2SeqEncoder.register("sparse_transformer")
class TransformerBlock(Seq2SeqEncoder):

    def __init__(self, emb, heads, ff_hidden_mult=4, dropout=0.0):
        super().__init__()
        
        self.input_dim = self.output_dim = emb

        self.attention = SelfAttentionNarrow(emb, heads=heads)

        self.norm1 = nn.LayerNorm(emb)
        self.norm2 = nn.LayerNorm(emb)

        self.ff = nn.Sequential(
            nn.Linear(emb, ff_hidden_mult * emb),
            nn.ReLU(),
            nn.Linear(ff_hidden_mult * emb, emb)
        )

        self.do = nn.Dropout(dropout)

    def forward(self, x, *args, **kwargs):
        attended = self.attention(x)
        
        x = self.norm1(attended + x)
        x = self.do(x)
        
        fedforward = self.ff(x)
        x = self.norm2(fedforward + x)
        x = self.do(x)

        return x

    @overrides
    def get_input_dim(self) -> int:
        return self.input_dim

    @overrides
    def get_output_dim(self) -> int:
        return self.output_dim

    @overrides
    def is_bidirectional(self) -> bool:
        return False