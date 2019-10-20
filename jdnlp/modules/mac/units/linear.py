# TODO: Find better place for this

from torch import nn
from torch.nn.init import kaiming_uniform_, xavier_uniform_, normal

def linear(in_dim, out_dim, bias=True):
    lin = nn.Linear(in_dim, out_dim, bias=bias)
    xavier_uniform_(lin.weight)
    if bias:
        lin.bias.data.zero_()

    return lin