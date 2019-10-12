# https://github.com/gaurav104/TextClassification/blob/master/Hierarchical%20Attention%20Network%20Text%20Classification.ipynb

from typing import Dict, Optional
from overrides import overrides

import torch
import torch.nn as nn

from allennlp.modules import TimeDistributed, Seq2VecEncoder, Seq2SeqEncoder

from jdnlp.modules.Attention.Attention import Attention
from jdnlp.modules.Attention.WordAttention import WordAttention
from jdnlp.modules.Attention.SentenceAttention import SentenceAttention
from jdnlp.modules.Attention.TimeDistributedAttention import TimeDistributedAttention

from copy import deepcopy

import logging
logger = logging.getLogger(__name__)

"""
input_size: int, 
hidden_size: int, 
attention_size: int,
n_layers: int = 1, 
dropout_p: float = 0.05, 
device="cpu
"""

@Seq2VecEncoder.register("HierarchicalAttention")
class HierarchialAttention(Seq2VecEncoder):
    def __init__(self, 
    word_encoder: Seq2SeqEncoder,
    sent_encoder: Seq2SeqEncoder
    ):
        super().__init__()

        """
        self.word_attention_model = WordAttention(input_size=input_size,
                                                  hidden_size=hidden_size,
                                                  attention_size=attention_size,
                                                  n_layers=n_layers,
                                                  dropout_p=dropout_p,
                                                  ).to(device)

        self.sentence_attention_model = SentenceAttention(input_size=hidden_size,
                                                          hidden_size=hidden_size,
                                                          attention_size=attention_size,
                                                          n_layers=n_layers,
                                                          dropout_p=dropout_p,
                                                          ).to(device)

        self.device = device
        """

        self.input_dim = word_encoder.get_input_dim()
        self.output_dim = sent_encoder.get_output_dim()

        self.word_encoder = TimeDistributed(word_encoder)
        self.sent_encoder = sent_encoder

        self.word_attn = Attention(word_encoder.get_output_dim())
        self.sent_attn = Attention(sent_encoder.get_output_dim())

    @overrides
    def forward(self, document: torch.Tensor, mask: torch.Tensor):
        """
        Parameters
        ----------
        document : Dict[str, Variable], required
            [B, N_SENT, N_WORDS, E_DIM]
        sentence_per_document : Dict[str, torch.Tensor], required
            The number of sentences for each document.
        word_per_sentence : Dict[str, torch.Tensor], required
            The number of words for each sentence in each document.
        label : Variable, optional (default = None)
            A variable representing the label for each instance in the batch.

        Returns
        -------
        An output dictionary consisting of:
        class_probabilities : torch.FloatTensor
            A tensor of shape ``(batch_size, num_classes)`` representing a distribution over the
            label classes for each instance.
        loss : torch.FloatTensor, optional
            A scalar loss to be optimised.
        """

        # mask is broken
        sentence_vecs = self.word_encoder(document, mask=mask)
        logger.warn(f"Word encodings: {sentence_vecs.size()}")
        
        sentence_vecs, word_weights = self.word_attn(sentence_vecs, return_attn_distribution=True) # self.word_attention_model(document, mask)
        logger.warn(f"Word-attn: {sentence_vecs.size()}")
        logger.warn(f"Word weights: {word_weights.size()}\n")

        doc_vecs = self.sent_encoder(sentence_vecs, mask=None)
        # doc_vecs = self.word_encoder(sentence_vecs.unsqueeze(0), mask=None)
        logger.warn(f"Sent encodings: {doc_vecs.size()}")

        doc_vecs, sentence_weights = self.sent_attn(sentence_vecs, return_attn_distribution=True) # sentence_attention_model , reduction_dim=-1
        logger.warn(f"Doc vecs: {doc_vecs.size()}")
        logger.warn(f"Sent weights: {sentence_weights.size()}")

        return doc_vecs, sentence_weights, word_weights


    @overrides
    def get_input_dim(self) -> int:
        # return self.word_attention_model.input_size
        return self.input_dim

    @overrides
    def get_output_dim(self) -> int:
        # return self.sentence_attention_model.hidden_size
        return self.output_dim

