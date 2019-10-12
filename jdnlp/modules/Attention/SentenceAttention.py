import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn.utils.rnn import pack_padded_sequence as pack
from torch.nn.utils.rnn import pad_packed_sequence as unpack
from torch.nn.utils.rnn import PackedSequence

from allennlp.modules import TimeDistributed

from allennlp.modules.similarity_functions import LinearSimilarity
from allennlp.modules.seq2vec_encoders import PytorchSeq2VecWrapper
from allennlp.modules.seq2seq_encoders import PytorchSeq2SeqWrapper

from jdnlp.modules.Attention.Attention import Attention

import logging
logger = logging.getLogger(__name__)

# nn.Module
class SentenceAttention(nn.Module):
    def __init__(self, input_size, hidden_size, n_layers=1, dropout_p=0):
        super().__init__()

        self.hidden_size = hidden_size
        self.rnn = nn.GRU(input_size=input_size,
                          hidden_size=hidden_size, #int(hidden_size / 2),
                          num_layers=n_layers,
                          dropout=dropout_p,
                          bidirectional=False,#True,
                          batch_first=True
                          )
        self.rnn = PytorchSeq2SeqWrapper(self.rnn)
        # self.encoder = TimeDistributed(self.rnn)
        self.attention_seq2seq = Attention(self.rnn.get_output_dim())
    
    def forward(self, sentences):
        # sentences = [batch_size, sentences, hidden_size]
        # sentences = sentences.unsqueeze(1)
        encoded = self.rnn(sentences, mask=None) # -> [batch_size, sentences, D]
        logger.warn(f"Doc vecs: {encoded.size()}")

        sent_weights, doc_vecs = self.attention_seq2seq(encoded, return_attn_distribution=True)
        logger.warn(f"Attn-Doc vecs: {doc_vecs.size()}")
        logger.warn(f"Sent attn: {sent_weights.size()}")

        return doc_vecs, sent_weights
    