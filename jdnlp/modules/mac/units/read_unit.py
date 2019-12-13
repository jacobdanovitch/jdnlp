import torch
from torch.autograd import Variable
from torch import nn
from torch.nn.init import kaiming_uniform_, xavier_uniform_, normal
import torch.nn.functional as F

from jdnlp.modules.mac.units.linear import linear
from entmax import sparsemax, entmax15


import torchsnooper


class ReadUnit(nn.Module):
    def __init__(self, dim, n_memories, save_attns=False):
        super().__init__()

        self.mem = linear(dim, dim)
        self.concat = linear(dim * 2, dim)
        self.attn = linear(dim, 1)
        
        
        self.saved_attn = []

    # @torchsnooper.snoop()
    def forward(self, memory, know, control):
        mem = self.mem(memory[-1]).unsqueeze(2)
        concat = self.concat(torch.cat([mem * know, know], 1).permute(0, 2, 1))
        attn = concat * control[-1].unsqueeze(1)
        attn = self.attn(attn).squeeze(2)
        
        attn = F.softmax(attn, 1)
        # attn = sparsemax(attn, 1)
        
        self.saved_attn.append(attn)
        
        attn = attn.unsqueeze(1)
        read = (attn * know).sum(2)

        return read
