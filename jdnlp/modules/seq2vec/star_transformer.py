"""
Full credit to:
https://github.com/fastnlp/fastNLP/blob/master/fastNLP/modules/encoder/star_transformer.py
"""


import torch
from torch import nn
import torch.nn.functional as F

import numpy as NP
from entmax import sparsemax, entmax_bisect

from typing import Dict, Optional
from overrides import overrides
from allennlp.modules import Seq2VecEncoder, Seq2SeqEncoder, TimeDistributed

import torchsnooper
import logging
logger = logging.getLogger(__name__)

# https://discuss.pytorch.org/t/how-to-make-the-parameter-of-torch-nn-threshold-learnable/4729/24
def ClampMax(x, val):
    """
    Clamps x to val.
    val >= 0.0
    """
    return x.clamp(min=0.0).sub(val).clamp(max=0.0).add(val) + x.clamp(max=0.0)

def ClampMin(x, val):
    """
    Clamps x to minimum value 'val'.
    val < 0.0
    """
    return x.clamp(max=0.0).sub(val).clamp(min=0.0).add(val) + x.clamp(min=0.0)

class AlphaEntmax(nn.Module):
    def __init__(self, nhead):
        super().__init__()

        self.nhead = nhead
        self.alpha = nn.Parameter(torch.randn(1, nhead).uniform_(1, 2)) # 1.5
        self.saved_alphas = []

    #@torchsnooper.snoop()#watch='self.alpha')
    def forward(self, x):
        self.alpha.data = ClampMax(ClampMin(self.alpha, 1), 2)
        alpha = self.alpha.repeat(x.size(0), 1).view(x.size(0), -1, *[1]*(x.dim()-2))#1, 1, 1)
        # self.saved_alphas.append(alpha.detach().cpu().numpy())
        return entmax_bisect(x, alpha=alpha, dim=3)#, n_iter=25) # ClampMax(self.alpha, 10)

@Seq2VecEncoder.register("star_transformer")
class StarTransformerPooling(Seq2VecEncoder):
    def __init__(self, hidden_size, num_layers, num_head, head_dim, unfold_size=3, dropout=0.1, max_len=None):
        super().__init__()
        self.star_transformer = StarTransformer(hidden_size, num_layers, num_head, head_dim, unfold_size=unfold_size, dropout=dropout, max_len=max_len)
        
        #relay_w = torch.randn(1).uniform_(0,1)
        #self.relay_w = nn.Parameter(relay_w)
        #self.nodes_w = nn.Parameter(1-relay_w)
    
    @overrides    
    def forward(self, x: torch.Tensor, mask: torch.Tensor=None, *args, **kwargs): 
        nodes, relay = self.star_transformer(x, mask)
        nodes = nodes.max(1)[0]
        return 0.5 * (relay + nodes)
        # return 0.5 * ((self.relay_w * relay) + (self.nodes_w * nodes))
        # return ((self.relay_w * relay) + (self.nodes_w * nodes)) / (self.relay_w + self.nodes_w)
    
    @overrides
    def get_input_dim(self) -> int:
        return self.star_transformer.get_input_dim()

    @overrides
    def get_output_dim(self) -> int:
        return self.star_transformer.get_output_dim()
    

@Seq2SeqEncoder.register("star_transformer") 
class StarTransformer(Seq2SeqEncoder):
    def __init__(self, hidden_size, num_layers, num_head, head_dim, unfold_size=3, dropout=0.1, max_len=None):
        super().__init__()
        self.input_dim = self.output_dim = hidden_size
        self.iters = num_layers

        self.norm = nn.ModuleList([nn.LayerNorm(hidden_size, eps=1e-6) for _ in range(self.iters)])
        self.emb_drop = nn.Dropout(dropout)
        self.ring_att = nn.ModuleList(
            [RingAttention(hidden_size, nhead=num_head, head_dim=head_dim, unfold_size=unfold_size, dropout=0.0)
             for _ in range(self.iters)])
        self.star_att = nn.ModuleList(
            [StarAttention(hidden_size, nhead=num_head, head_dim=head_dim, unfold_size=unfold_size, dropout=0.0)
             for _ in range(self.iters)])

        self.pos_emb = max_len and nn.Embedding(max_len, hidden_size)

    # @torchsnooper.snoop(watch='nodes.size()')# @overrides
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
            # nodes = norm_func(self.norm[i], nodes)
            nodes = self.ring_att[i](nodes, ax=ax)
            # nodes = F.leaky_relu(nodes)
            relay = self.star_att[i](relay, torch.cat([relay, nodes], 2), smask)
            # relay = F.leaky_relu(relay)
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


class RingAttention(nn.Module):
    def __init__(self, nhid, nhead=10, head_dim=10, unfold_size=3, dropout=0.1):
        super().__init__()
        # Multi-head Self Attention Case 1, doing self-attention for small regions
        # Due to the architecture of GPU, using hadamard production and summation are faster than dot production when unfold_size is very small
        self.WQ = nn.Conv2d(nhid, nhead*head_dim, 1)
        self.WK = nn.Conv2d(nhid, nhead*head_dim, 1)
        self.WV = nn.Conv2d(nhid, nhead*head_dim, 1)
        self.WO = nn.Conv2d(nhead * head_dim, nhid, 1)

        self.drop = nn.Dropout(dropout)
        self.nhid, self.nhead, self.head_dim, self.unfold_size = nhid, nhead, head_dim, unfold_size

        self.softmax = AlphaEntmax(nhead)

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
        k = F.unfold(k, (unfold_size, 1), padding=(unfold_size // 2, 0)) \
            .view(B, nhead, head_dim, unfold_size, L)
        v = F.unfold(v, (unfold_size, 1), padding=(unfold_size // 2, 0)) \
            .view(B, nhead, head_dim, unfold_size, L)
        if ax is not None:
            k = torch.cat([k, ak], 3)
            v = torch.cat([v, av], 3)

        scale = NP.sqrt(head_dim)
        alphas = (q * k).sum(2, keepdim=True) / scale
        alphas = self.softmax(alphas)
        alphas = self.drop(alphas)  # B N L 1 U
        att = (alphas * v).sum(3).view(B, nhead * head_dim, L, 1)
        ret = self.WO(att)
        return ret
        
        
class StarAttention(nn.Module):
    def __init__(self, nhid, nhead=10, head_dim=10, unfold_size=3, dropout=0.1):
        # Multi-head Self Attention Case 2, a broadcastable query for a sequence key and value
        super().__init__()
        self.WQ = nn.Conv2d(nhid, nhead * head_dim, 1)
        self.WK = nn.Conv2d(nhid, nhead * head_dim, 1)
        self.WV = nn.Conv2d(nhid, nhead * head_dim, 1)
        self.WO = nn.Conv2d(nhead * head_dim, nhid, 1)

        self.drop = nn.Dropout(dropout)
        self.nhid, self.nhead, self.head_dim, self.unfold_size = nhid, nhead, head_dim, unfold_size

        self.softmax = AlphaEntmax(nhead)

    def forward(self, x, y, mask=None):
        # x: B, H, 1, 1, 1 y: B H L 1
        nhid, nhead, head_dim, unfold_size = self.nhid, self.nhead, self.head_dim, self.unfold_size
        B, H, L, _ = y.shape

        q, k, v = self.WQ(x), self.WK(y), self.WV(y)

        q = q.view(B, nhead, 1, head_dim)  # B, H, 1, 1 -> B, N, 1, h
        k = k.view(B, nhead, head_dim, L)  # B, H, L, 1 -> B, N, h, L
        v = v.view(B, nhead, head_dim, L).permute(0, 1, 3, 2)  # B, H, L, 1 -> B, N, L, h
        
        scale = NP.sqrt(head_dim)
        pre_a = torch.matmul(q, k) / scale
        if mask is not None:
            pre_a = pre_a.masked_fill(mask[:, None, None, :], -float(100))
        alphas = self.softmax(pre_a)
        alphas = self.drop(alphas) # B, N, 1, L
        att = torch.matmul(alphas, v).view(B, -1, 1, 1)  # B, N, 1, h -> B, N*h, 1, 1
        return self.WO(att)

class ConcatAttentionHeads(nn.Module):
    def __init__(self, Attention, nhid, nhead=10, head_dim=10, dropout=0.1):
        super().__init__()
        assert nhid == nhead * head_dim
        self.attn_heads = nn.ModuleList([Attention(nhid, head_dim, dropout) for _ in range(nhead)])
        self.WO = nn.Conv2d(nhead * head_dim, nhid, 1)
    
    def forward(self, x, *args, **kwargs):
        #concat = [att(x, *args, **kwargs) for att in self.attn_heads]
        #"""
        concat = []
        streams = [(head, torch.cuda.Stream()) for head in self.attn_heads]
        torch.cuda.synchronize()
        
        for head, stream in streams:
            with torch.cuda.stream(stream):
                concat.append(head(x, *args, **kwargs))
        torch.cuda.synchronize()
        # """
        
        proj = self.WO(torch.cat(concat, dim=1))
        return proj
    
    
class RingAttentionHead(nn.Module):
    def __init__(self, nhid, head_dim=10, dropout=0.1):
        super().__init__()
        # Multi-head Self Attention Case 1, doing self-attention for small regions
        # Due to the architecture of GPU, using hadamard production and summation are faster than dot production when unfold_size is very small
        self.WQ = nn.Conv2d(nhid, head_dim, 1)
        self.WK = nn.Conv2d(nhid, head_dim, 1)
        self.WV = nn.Conv2d(nhid, head_dim, 1)

        self.drop = nn.Dropout(dropout)
        self.nhid, self.nhead, self.head_dim, self.unfold_size = nhid, 1, head_dim, 3

        self.softmax = AlphaEntmax()

    # @torchsnooper.snoop()
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

        scale = NP.sqrt(head_dim)
        alphas = (q * k).sum(2, keepdim=True) / scale
        # alphas = self.softmax(alphas, alpha=ClampMax(self.alpha, 10), dim=3)
        alphas = self.softmax(alphas)
        alphas = self.drop(alphas)  # B N L 1 U
        att = (alphas * v).sum(3).view(B, nhead * head_dim, L, 1)
        return att
        # ret = self.WO(att)
        # return ret
        
        
class StarAttentionHead(nn.Module):
    def __init__(self, nhid, head_dim=10, dropout=0.1):
        # Multi-head Self Attention Case 2, a broadcastable query for a sequence key and value
        super().__init__()
        self.WQ = nn.Conv2d(nhid, head_dim, 1)
        self.WK = nn.Conv2d(nhid, head_dim, 1)
        self.WV = nn.Conv2d(nhid, head_dim, 1)

        self.drop = nn.Dropout(dropout)
        self.nhid, self.nhead, self.head_dim, self.unfold_size = nhid, 1, head_dim, 3

        self.softmax = AlphaEntmax()

    def forward(self, x, y, mask=None):
        # x: B, H, 1, 1, 1 y: B H L 1
        nhid, nhead, head_dim, unfold_size = self.nhid, self.nhead, self.head_dim, self.unfold_size
        B, H, L, _ = y.shape

        q, k, v = self.WQ(x), self.WK(y), self.WV(y)

        q = q.view(B, nhead, 1, head_dim)  # B, H, 1, 1 -> B, N, 1, h
        k = k.view(B, nhead, head_dim, L)  # B, H, L, 1 -> B, N, h, L
        v = v.view(B, nhead, head_dim, L).permute(0, 1, 3, 2)  # B, H, L, 1 -> B, N, L, h
        
        scale = NP.sqrt(head_dim)
        pre_a = torch.matmul(q, k) / scale
        if mask is not None:
            pre_a = pre_a.masked_fill(mask[:, None, None, :], -float(100))
        alphas = self.softmax(pre_a)
        alphas = self.drop(alphas)
        # alphas = self.drop(self.softmax(pre_a, alpha=ClampMax(self.alpha,10), dim=3))  # B, N, 1, L
        att = torch.matmul(alphas, v).view(B, -1, 1, 1)  # B, N, 1, h -> B, N*h, 1, 1
        return att