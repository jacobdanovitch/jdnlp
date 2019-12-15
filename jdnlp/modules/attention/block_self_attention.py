from overrides import overrides

import torch
import torch.nn as nn
import torch.nn.functional as F

from allennlp.modules import FeedForward, Seq2VecEncoder, Seq2SeqEncoder
from entmax import sparsemax

import torchsnooper
import logging
logger = logging.getLogger(__name__)

# softmax_fn = sparsemax # 
softmax_fn = F.softmax

def calculate_block_size(expected_max_len):
    return int((2 * expected_max_len) ** (1/3))


class s2tSA(nn.Module):
    def __init__(self, hidden_size):
        super(s2tSA, self).__init__()
        
        self.s2t_W1 = nn.Sequential(nn.Linear(hidden_size, hidden_size), nn.ReLU())
        self.s2t_W = nn.Linear(hidden_size, hidden_size)

    def forward(self, x):
        """
        source2token self-attention module
        :param x: (batch, (block_num), seq_len, hidden_size)
        :return: s: (batch, (block_num), hidden_size)
        """
        # (batch, (block_num), seq_len, word_dim)
        f = self.s2t_W1(x)
        f = softmax_fn(self.s2t_W(f), dim=-2)
        # (batch, (block_num), word_dim)
        s = torch.sum(f * x, dim=-2)
        return s

class mBloSA(nn.Module):
    def __init__(self, 
                input_dim: int, 
                block_size: int,
                mSA_scalar: int = 5.0, 
                mask='fw'
        ):
        super().__init__()

        self.word_dim = input_dim
        self.mSA_scalar = mSA_scalar
        self.block_size = block_size
        
        self.mask = mask
        
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        logger.warn(f'Initializing on device: {self.device}')

        # init submodules
        self.s2tSA = s2tSA(self.word_dim)
        self.init_mSA()
        self.init_mBloSA()

    
    def init_mSA(self):
        self.m_W1 = nn.Linear(self.word_dim, self.word_dim, bias=False)
        self.m_W2 = nn.Linear(self.word_dim, self.word_dim, bias=False)
        self.m_b = nn.Parameter(torch.zeros(self.word_dim))

        self.c = nn.Parameter(torch.Tensor([self.mSA_scalar]), requires_grad=False)

    def init_mBloSA(self):
        self.g_W1 = nn.Linear(self.word_dim, self.word_dim, bias=False)
        self.g_W2 = nn.Linear(self.word_dim, self.word_dim, bias=False)
        self.g_b = nn.Parameter(torch.zeros(self.word_dim))

        self.f_W1 = nn.Sequential(nn.Linear(self.word_dim * 3, self.word_dim), nn.ReLU())
        self.f_W2 = nn.Linear(self.word_dim * 3, self.word_dim)

    def mSA(self, x):
        """
        masked self-attention module
        :param x: (batch, (block_num), seq_len, word_dim)
        :return: s: (batch, (block_num), seq_len, word_dim)
        """
        seq_len = x.size(-2)

        # (batch, (block_num), seq_len, 1, word_dim)
        x_i = self.m_W1(x).unsqueeze(-2)
        # (batch, (block_num), 1, seq_len, word_dim)
        x_j = self.m_W2(x).unsqueeze(-3)

        # build fw or bw masking
        # (seq_len, seq_len)
        M = torch.ones((seq_len, seq_len)).cuda().triu().detach()
        M[M == 1] = float('-inf')

        # CASE 1 - x: (batch, seq_len, word_dim)
        # (1, seq_len, seq_len, 1)
        M = M.contiguous().view(1, M.size(0), M.size(1), 1)
        # (batch, 1, seq_len, word_dim)
        # padding to deal with nan
        pad = torch.zeros(x.size(0), 1, x.size(-2), x.size(-1))
        pad = pad.cuda().detach()

        # CASE 2 - x: (batch, block_num, seq_len, word_dim)
        if len(x.size()) == 4:
            M = M.unsqueeze(1)
            pad = torch.stack([pad] * x.size(1), dim=1)

        # (batch, (block_num), seq_len, seq_len, word_dim)
        f = self.c * torch.tanh((x_i + x_j + self.m_b) / self.c)

        # fw or bw masking
        if f.size(-2) > 1:
            if self.mask == 'fw':
                M = M.transpose(-2, -3)
                f = softmax_fn((f + M).narrow(-3, 0, f.size(-3) - 1), dim=-2)
                f = torch.cat([f, pad], dim=-3)
            elif self.mask == 'bw':
                f = softmax_fn((f + M).narrow(-3, 1, f.size(-3) - 1), dim=-2)
                f = torch.cat([pad, f], dim=-3)
            else:
                raise NotImplementedError('only fw or bw mask is allowed!')
        else:
            f = pad

        # (batch, (block_num), seq_len, word_dim)
        s = torch.sum(f * x.unsqueeze(-2), dim=-2)
        return s
    
    def forward(self, x):
        """
        masked block self-attention module
        :param x: (batch, seq_len, word_dim)
        :param M: (seq_len, seq_len)
        :return: (batch, seq_len, word_dim)
        """
        r = self.block_size
        n = x.size(1)
        m = n // r

        # padding for the same length of each block
        pad_len = (r - n % r) % r
        if pad_len:
            pad = torch.zeros(x.size(0), pad_len, x.size(2)).to(self.device).detach()
            x = torch.cat([x, pad], dim=1)

        # --- Intra-block self-attention ---
        # x = (batch, block_num(m), seq_len(r), word_dim)
        x = torch.stack([x.narrow(1, i, r) for i in range(0, x.size(1), r)], dim=1)
        # h = (batch, block_num(m), seq_len(r), word_dim)
        h = self.mSA(x)
        # v = (batch, block_num(m), word_dim)
        v = self.s2tSA(h)

        # --- Inter-block self-attention ---
        # (batch, m, word_dim)
        o = self.mSA(v)
        # (batch, m, word_dim)
        G = torch.sigmoid(self.g_W1(o) + self.g_W2(v) + self.g_b)
        # (batch, m, word_dim)
        e = G * o + (1 - G) * v

        # --- Context fusion ---
        # (batch, n, word_dim)
        E = torch.cat([torch.stack([e.select(1, i)] * r, dim=1) for i in range(e.size(1))], dim=1).narrow(1, 0, n)
        x = x.view(x.size(0), -1, x.size(-1)).narrow(1, 0, n)
        h = h.view(h.size(0), -1, h.size(-1)).narrow(1, 0, n)

        # (batch, n, word_dim * 3) -> (batch, n, word_dim)
        fusion = self.f_W1(torch.cat([x, h, E], dim=2))
        G = torch.sigmoid(self.f_W2(torch.cat([x, h, E], dim=2)))
        # (batch, n, word_dim)
        u = G * fusion + (1 - G) * x

        return u
  
  
  
@Seq2SeqEncoder.register('block-self-attention')
class BiBloSAN(Seq2SeqEncoder):
    def __init__(self, 
                 input_dim: int, 
                 expected_max_length: int = 100, 
                 mSA_scalar: int = 5.0
        ):
        super().__init__()
        
        self.word_dim = word_dim = input_dim
        
        block_size = calculate_block_size(expected_max_length)
        self.mBloSA_fw = mBloSA(word_dim, block_size, mSA_scalar, mask='fw')
        self.mBloSA_bw = mBloSA(word_dim, block_size, mSA_scalar, mask='bw')

        # two untied fully connected layers
        self.fc_fw = nn.Sequential(nn.Linear(word_dim, word_dim), nn.ReLU())
        self.fc_bw = nn.Sequential(nn.Linear(word_dim, word_dim), nn.ReLU())

        self.proj = nn.Sequential(nn.Linear(word_dim*2, word_dim), nn.ReLU())

    def forward(self, x, *args, **kwargs):
        input_fw = self.fc_fw(x)
        input_bw = self.fc_bw(x)

        # (batch, seq_len, word_dim)
        u_fw = self.mBloSA_fw(input_fw)
        u_bw = self.mBloSA_bw(input_bw)

        # (batch, seq_len, word_dim * 2) -> (batch, word_dim * 2)
        u_bi = torch.cat([u_fw, u_bw], dim=2)
        u_bi = self.proj(u_bi)
        return u_bi

    @overrides
    def get_input_dim(self):
        return self.word_dim
    
    @overrides
    def get_output_dim(self):
        return self.word_dim
    
    @overrides
    def is_bidirectional(self):
        return True