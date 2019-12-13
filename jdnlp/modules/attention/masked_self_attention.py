from typing import Dict, Optional
from overrides import overrides

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn import Parameter, Linear

from allennlp.nn import util
from allennlp.modules import TimeDistributed, Seq2VecEncoder, Seq2SeqEncoder

import logging
logger = logging.getLogger(__name__)

@Seq2VecEncoder.register("masked_self_attention")
class MaskedSelfAttention(Seq2VecEncoder):
    def __init__(self, encoder: Seq2SeqEncoder, time_distributed=False):
        super().__init__()
        self.encoder = TimeDistributed(encoder) if time_distributed else encoder
        self.attn_projection = nn.Linear(encoder.get_input_dim(), 1)
        
    def forward(self, x: torch.Tensor, mask: torch.Tensor):
        encoded = self.encoder(x, mask)
        attn_logits = self.attn_projection(encoded).squeeze(-1)
        attn_weights = util.masked_softmax(attn_logits, mask)
        attn_output = util.weighted_sum(encoded, attn_weights)
        
        return attn_output, attn_weights
    
    @overrides
    def get_input_dim(self) -> int:
        return self.encoder.get_input_dim()

    @overrides
    def get_output_dim(self) -> int:
        return self.encoder.get_output_dim()