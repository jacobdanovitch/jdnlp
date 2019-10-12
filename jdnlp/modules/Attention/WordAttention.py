import torch
import torch.nn as nn
# import torch.nn.functional as F


from allennlp.modules.seq2vec_encoders import PytorchSeq2VecWrapper
from allennlp.modules.seq2seq_encoders import PytorchSeq2SeqWrapper
from allennlp.modules import TimeDistributed
from allennlp.nn import util
from allennlp.modules.attention import LinearAttention

from jdnlp.modules.Attention.Attention import Attention

import logging
logger = logging.getLogger(__name__)

class WordAttention(nn.Module):
    def __init__(self, input_size, hidden_size, attention_size, n_layers=1, dropout_p=0):
        super().__init__()

        self.input_size = input_size
        self.rnn = nn.GRU(input_size=input_size,
                          hidden_size=hidden_size, #int(hidden_size / 2),
                          num_layers=n_layers,
                          dropout=dropout_p,
                          bidirectional=False,#True,
                          batch_first=True
                          )
        # self.rnn = PytorchSeq2VecWrapper(self.rnn)
        self.rnn = PytorchSeq2SeqWrapper(self.rnn)
        self.encoder = TimeDistributed(self.rnn)
        self.attention_seq2seq = Attention(self.rnn.get_output_dim())


    def forward(self, document, mask):
        # B, N_S, N_W, D = document.size()
        # words = document.view(B*N_S, N_W, D)
        # logger.warn(f"Words: {words.size()}")

        # mask is wrong
        encoded = self.encoder(document, mask=None)  # -> [B, N_S, N_W, D]
        logger.warn(f"Encoded: {encoded.size()}")

        word_weights, sent_vecs = self.attention_seq2seq(encoded, return_attn_distribution=True) # -> [B, N_S, D]
        
        # encoded = encoded.view(B, N_S, D)
        # attn_dist = attn_dist.view(B, N_S, D)

        logger.warn(f"Sent vecs: {sent_vecs.size()}")
        # logger.warn(f"Word attn: {word_weights.size()}")

        return sent_vecs, word_weights
