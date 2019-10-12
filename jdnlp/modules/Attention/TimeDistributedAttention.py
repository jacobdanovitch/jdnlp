import torch
import torch.nn as nn

from allennlp.modules.seq2vec_encoders import PytorchSeq2VecWrapper
from allennlp.modules.seq2seq_encoders import PytorchSeq2SeqWrapper, Seq2SeqEncoder
from allennlp.modules import TimeDistributed
from allennlp.nn import util

from jdnlp.modules.Attention.Attention import Attention

import logging
logger = logging.getLogger(__name__)

class TimeDistributedAttention(nn.Module):
    def __init__(self, encoder: Seq2SeqEncoder):
        super().__init__()

        self.encoder = TimeDistributed(encoder)
        self.attention_seq2seq = Attention(encoder.get_output_dim())


    def forward(self, document, mask=None, **kwargs):
        encoded = self.encoder(document, mask=mask)
        encoded.squeeze()
        # logging.critical(f"Encoded: {encoded.size()}")
        weights, vecs = self.attention_seq2seq(encoded, return_attn_distribution=True, **kwargs)
        # logging.critical(f"Vecs: {vecs.size()}")

        return vecs, weights
