# https://github.com/gaurav104/TextClassification/blob/master/Hierarchical%20Attention%20Network%20Text%20Classification.ipynb

from typing import Dict, Optional
from overrides import overrides

import torch
import torch.nn as nn

from allennlp.modules import TimeDistributed, Seq2VecEncoder, Seq2SeqEncoder

from jdnlp.modules.Attention.Attention import Attention
# from jdnlp.utils import compare

from copy import deepcopy

import logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.CRITICAL)

@Seq2VecEncoder.register("HierarchicalAttention")
class HierarchialAttention(Seq2VecEncoder):
    def __init__(self, 
    word_encoder: Seq2SeqEncoder,
    sent_encoder: Seq2SeqEncoder):
        super().__init__()

        self.input_dim = word_encoder.get_input_dim()
        self.output_dim = sent_encoder.get_output_dim()

        self.word_encoder = TimeDistributed(word_encoder)
        self.sent_encoder = sent_encoder.train()
        self.sent_encoder.requires_grad = True

        # self.word_attn = Attention(self.input_dim)
        # self.sent_attn = Attention(self.output_dim)
        
        self.sent_enc_grad = 0
        self.params = None
        # logging.critical([(n, p.requires_grad) for (n, p) in self.sent_encoder.named_parameters()])
        # logging.critical(self.sent_encoder)
        

    @overrides
    def forward(self, document: torch.Tensor, mask: torch.Tensor):
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

        # mask is broken
        
        sentence_weights, word_weights = None, None
        
        # sentence_vecs = document.sum(dim=1)
        sentence_vecs = self.word_encoder(document, mask=None).sum(dim=1)
        # logger.warn(f"Word encodings: {sentence_vecs.size()}")
        
        # sentence_vecs, word_weights = self.word_attn(sentence_vecs, return_attn_distribution=True) # self.word_attention_model(document, mask)
        # logger.warn(f"Word-attn: {sentence_vecs.size()}")
        # logger.warn(f"Word weights: {word_weights.size()}\n")

        # doc_vecs = sentence_vecs.sum(dim=1)

        doc_vecs = self.sent_encoder(sentence_vecs, mask=None).sum(dim=1)
        # doc_vecs = self.word_encoder(sentence_vecs.unsqueeze(0), mask=None)
        #logger.warn(f"Sent encodings: {doc_vecs.size()}")

        # doc_vecs, sentence_weights = self.sent_attn(sentence_vecs, return_attn_distribution=True) # sentence_attention_model , reduction_dim=-1
        # logger.warn(f"Doc vecs: {doc_vecs.size()}")
        # logger.warn(f"Sent weights: {sentence_weights.size()}")
        # logger.critical(list(self.word_encoder.parameters())[0])
        
        # assert False
        

        return doc_vecs, sentence_weights, word_weights


    @overrides
    def get_input_dim(self) -> int:
        # return self.word_attention_model.input_size
        return self.input_dim

    @overrides
    def get_output_dim(self) -> int:
        # return self.sentence_attention_model.hidden_size
        return self.output_dim



