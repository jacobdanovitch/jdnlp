import torch
import torch.nn as nn
# import torch.nn.functional as F


from allennlp.modules.seq2vec_encoders import PytorchSeq2VecWrapper
from allennlp.modules.seq2seq_encoders import PytorchSeq2SeqWrapper
from allennlp.nn import util
from allennlp.modules.attention import LinearAttention

from jdnlp.modules.Attention.Attention import Attention

import logging
logger = logging.getLogger(__name__)

class WordAttention(nn.Module):
    def __init__(self, input_size, hidden_size, attention_size, n_layers=1, dropout_p=0, device="cpu"):
        super().__init__()

        self.device=device
        self.input_size = input_size
        self.rnn = nn.GRU(input_size=input_size,
                          hidden_size=hidden_size, #int(hidden_size / 2),
                          num_layers=n_layers,
                          dropout=dropout_p,
                          bidirectional=False,#True,
                          batch_first=True
                          )
        # self.rnn = PytorchSeq2VecWrapper(self.rnn)
        self.encoder = PytorchSeq2SeqWrapper(self.rnn)
        self.attention_seq2seq = Attention(self.encoder.get_output_dim())


    def forward(self, document, mask):
        encoded = self.encoder(document, mask)
        logger.warn(f"Encoded: {encoded.size()}")

        attn_dist, encoded = self.attention_seq2seq(encoded, return_attn_distribution=True)
        logger.warn(f"Attn: {attn_dist.size()}")
