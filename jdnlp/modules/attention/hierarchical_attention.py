# https://github.com/gaurav104/TextClassification/blob/master/Hierarchical%20Attention%20Network%20Text%20Classification.ipynb

from typing import Dict, Optional
from overrides import overrides

import torch
import torch.nn as nn

from allennlp.modules import TimeDistributed, Seq2VecEncoder, Seq2SeqEncoder
from allennlp.nn import util

from jdnlp.modules.attention.masked_self_attention import MaskedSelfAttention

import logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.CRITICAL)

@Seq2VecEncoder.register("HierarchicalAttention")
class HierarchialAttention(Seq2VecEncoder):
    def __init__(self, 
    word_encoder: Seq2SeqEncoder,
    sent_encoder: Seq2SeqEncoder
    ):
        super().__init__()

        self.input_dim = word_encoder.get_input_dim()
        self.output_dim = sent_encoder.get_output_dim()

        self.word_attn = MaskedSelfAttention(word_encoder, time_distributed=True)
        self.sent_attn = MaskedSelfAttention(sent_encoder)
    
    @overrides
    def forward(self, document: torch.Tensor, word_mask: torch.Tensor, sent_mask: torch.Tensor):
        """
        Parameters
        ----------
        document : Dict[str, Variable], required
            [B, N_SENT, N_WORDS, E_DIM]
        label : Variable, optional (default = None)
            A variable representing the label for each instance in the batch.

        Returns
        -------
        """
        
        sentence_vecs, word_weights = self.word_attn(document, word_mask)
        doc_vecs, sentence_weights = self.sent_attn(sentence_vecs, sent_mask)

        return doc_vecs, sentence_weights, word_weights


    @overrides
    def get_input_dim(self) -> int:
        return self.input_dim

    @overrides
    def get_output_dim(self) -> int:
        return self.output_dim



