import torch
from torch import nn
import torch.nn.functional as F

from entmax import sparsemax

import random, math

from typing import Dict, Optional
from overrides import overrides
from allennlp.modules import Seq2VecEncoder, Seq2SeqEncoder, TimeDistributed

import logging
logger = logging.getLogger(__name__)

@Seq2VecEncoder.register("pooling")
class PoolingEncoder(Seq2VecEncoder):
    def __init__(self, 
                 encoder: Seq2SeqEncoder,
                 op: str = 'sum', # TODO: change this to registerable
                 time_distributed: bool = False
        ):
        super().__init__()
        
        self.input_dim = encoder.get_input_dim()
        self.output_dim = encoder.get_output_dim()
        
        self.encoder = TimeDistributed(encoder) if time_distributed else encoder
        self.op = getattr(torch, op)
        
    def forward(self, x: torch.Tensor, mask: torch.Tensor=None, dim:int =-2, *args, **kwargs): 
        x = self.encoder(x, mask)
        return self.op(x, dim=dim)
    
    
    @overrides
    def get_input_dim(self) -> int:
        return self.input_dim

    @overrides
    def get_output_dim(self) -> int:
        return self.output_dim