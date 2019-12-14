"""
Full credit to:
https://github.com/serrano-s/attn-tests/blob/master/attn_tests_lib/simple_han_attn_layer.py
"""

from overrides import overrides
from typing import Dict, Optional, Callable

from allennlp.modules import Seq2SeqEncoder, Seq2VecEncoder, TimeDistributed
from allennlp.nn.util import get_text_field_mask, masked_softmax
from allennlp.nn import util
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import numpy as np

from entmax import sparsemax, entmax_bisect

import torchsnooper
import logging
logger = logging.getLogger(__name__)

"""
NOTE: Softmax -> Sparsemax
"""

def masked_activation(
    fn: Callable,
    vector: torch.Tensor,
    mask: torch.Tensor,
    dim: int = -1,
    memory_efficient: bool = False,
    mask_fill_value: float = -1e32,
) -> torch.Tensor:
    """
    ``torch.nn.functional.softmax(vector)`` does not work if some elements of ``vector`` should be
    masked.  This performs a softmax on just the non-masked portions of ``vector``.  Passing
    ``None`` in for the mask is also acceptable; you'll just get a regular softmax.
    ``vector`` can have an arbitrary number of dimensions; the only requirement is that ``mask`` is
    broadcastable to ``vector's`` shape.  If ``mask`` has fewer dimensions than ``vector``, we will
    unsqueeze on dimension 1 until they match.  If you need a different unsqueezing of your mask,
    do it yourself before passing the mask into this function.
    If ``memory_efficient`` is set to true, we will simply use a very large negative number for those
    masked positions so that the probabilities of those positions would be approximately 0.
    This is not accurate in math, but works for most cases and consumes less memory.
    In the case that the input vector is completely masked and ``memory_efficient`` is false, this function
    returns an array of ``0.0``. This behavior may cause ``NaN`` if this is used as the last layer of
    a model that uses categorical cross-entropy loss. Instead, if ``memory_efficient`` is true, this function
    will treat every element as equal, and do softmax over equal numbers.
    """
    if mask is None:
        result = fn(vector, dim=dim)
    else:
        mask = mask.float()
        while mask.dim() < vector.dim():
            mask = mask.unsqueeze(1)
        if not memory_efficient:
            # To limit numerical errors from large vector elements outside the mask, we zero these out.
            result = fn(vector * mask, dim=dim)
            result = result * mask
            result = result / (result.sum(dim=dim, keepdim=True) + 1e-13)
        else:
            masked_vector = vector.masked_fill((1 - mask).to(dtype=torch.bool), mask_fill_value)
            result = fn(masked_vector, dim=dim)
    return result

@Seq2SeqEncoder.register("simple_han_attention")
class SimpleHanAttention(Seq2SeqEncoder):
    def __init__(self,
                 input_dim : int = None,
                 context_vector_dim: int = None) -> None:
        super().__init__()
        context_vector_dim = context_vector_dim or input_dim
        
        #self.alpha = torch.nn.Parameter(torch.randn(1))
        
        self._mlp = torch.nn.Linear(input_dim, context_vector_dim, bias=True)
        self._context_dot_product = torch.nn.Linear(context_vector_dim, 1, bias=False)
        self.vec_dim = self._mlp.weight.size(1)
        
        self._encoder = Seq2SeqEncoder.by_name('gru')(
            input_size=input_dim,
            hidden_size=input_dim//2,
            bidirectional=True
        )

    @overrides
    def get_input_dim(self) -> int:
        return self.vec_dim

    @overrides
    def get_output_dim(self) -> int:
        return self.vec_dim

    @overrides
    def is_bidirectional(self):
        return False

    # @torchsnooper.snoop() # @overrides
    def forward(self, tokens: torch.Tensor, mask: torch.Tensor):  # pylint: disable=arguments-differ
        assert mask is not None
        batch_size, sequence_length, embedding_dim = tokens.size()
        
        tokens = self._encoder(tokens, mask)

        attn_weights = tokens.view(batch_size * sequence_length, embedding_dim)
        attn_weights = torch.tanh(self._mlp(attn_weights))
        attn_weights = self._context_dot_product(attn_weights)
        attn_weights = attn_weights.view(batch_size, -1)  # batch_size x seq_len
        
        attn_weights = F.softmax(attn_weights, dim=-1)
        # attn_weights = entmax_bisect(attn_weights, alpha=self.alpha, dim=-1)
        # attn_weights = sparsemax(attn_weights, dim=-1) # 
        # attn_weights = masked_activation(lambda x, dim: (entmax_bisect(x, alpha=self.alpha, dim=dim)), attn_weights, mask)
        
        attn_weights = attn_weights.unsqueeze(2).expand(batch_size, sequence_length, embedding_dim)

        return tokens * attn_weights
    
@Seq2SeqEncoder.register("stacked_han_attention")
class MultiHeadDotAttention(Seq2SeqEncoder):
    def __init__(self,
                 input_dim : int = None,
                 context_vector_dim: int = None,
                 n_heads: int = 1
                 ) -> None:
        super().__init__()
        
        self.attn_heads = nn.ModuleList([SimpleHanAttention(input_dim, context_vector_dim) for _ in range(n_heads)])
        self.proj = nn.Linear(input_dim * n_heads, input_dim)

    @overrides
    def get_input_dim(self) -> int:
        return self.attn_heads[0].get_input_dim()

    @overrides
    def get_output_dim(self) -> int:
        return self.attn_heads[0].get_output_dim()

    @overrides
    def is_bidirectional(self):
        return False

    # @torchsnooper.snoop() # @overrides
    def forward(self, tokens: torch.Tensor, mask: torch.Tensor):
        x = torch.cat([head(tokens, mask) for head in self.attn_heads], dim=-1)
        x = self.proj(x)
        return x
    

@Seq2VecEncoder.register("han_encoder")
class HanEncoder(Seq2VecEncoder):
    """
    This ``Model`` implements the Hierarchical Attention Network described in
    https://www.semanticscholar.org/paper/Hierarchical-Attention-Networks-for-Document-Yang-Yang/1967ad3ac8a598adc6929e9e6b9682734f789427
    by Yang et. al, 2016.
    Parameters
    ----------
    word_encoder : ``Seq2SeqEncoder``
        Used to encode words.
    sentence_encoder : ``Seq2SeqEncoder``
        Used to encode sentences.
    word_attention : ``Seq2VecEncoder``
        Seq2Vec layer that (in original implementation) uses attention to calculate a fixed-length vector
        representation of each sentence from that sentence's sequence of word vectors
    sentence_attention : ``Seq2VecEncoder``
        Seq2Vec layer that (in original implementation) uses attention to calculate a fixed-length vector
        representation of each document from that document's sequence of sentence vectors
    classification_layer : ``FeedForward``
        This feedforward network computes the output logits.
    pre_word_encoder_dropout : ``float``, optional (default=0.0)
        Dropout percentage to use before word_attention encoder.
    pre_sentence_encoder_dropout : ``float``, optional (default=0.0)
        Dropout percentage to use before sentence_attention encoder.
    """
    def __init__(self,
                 sentence_encoder: Seq2SeqEncoder,
                 document_encoder: Seq2SeqEncoder,
                 word_attention: Seq2SeqEncoder,
                 sentence_attention: Seq2SeqEncoder,
                 pre_sentence_encoder_dropout: float = 0.0,
                 pre_document_encoder_dropout: float = 0.0
    ):
        super().__init__()
        
        self.input_dim = sentence_encoder.get_input_dim()
        self.output_dim = document_encoder.get_output_dim()

        self._word_attention = word_attention
        self._sentence_attention = sentence_attention
        # self._pre_sentence_encoder_dropout = torch.nn.Dropout(p=pre_sentence_encoder_dropout)
        self._sentence_encoder = sentence_encoder # TimeDistributed()
        # self._pre_document_encoder_dropout = torch.nn.Dropout(p=pre_document_encoder_dropout)
        self._document_encoder = document_encoder

    @overrides
    def forward(self,  # type: ignore
                embedded_words: torch.Tensor,
                mask: torch.Tensor,
                sentence_level_mask: torch.Tensor
                ) -> torch.Tensor:
        """
        Parameters
        ----------
        tokens : Dict[str, torch.LongTensor]
            From a ``TextField``. These are tokens should be segmented into their respective sentences.
        Returns
        -------
        An output dictionary consisting of:
        label_logits : torch.FloatTensor
            A tensor of shape ``(batch_size, num_labels)`` representing unnormalised log
            probabilities of the label.
        label_probs : torch.FloatTensor
            A tensor of shape ``(batch_size, num_labels)`` representing probabilities of the
            label.
        loss : torch.FloatTensor, optional
            A scalar loss to be optimised.
        """
        # 4-d tensor: all_docs x max_num_sents_in_any_doc x max_num_tokens_in_any_doc_sent x dim
        # these embeddings  should be sentence-segmented.
        batch_size, max_num_sents, max_num_tokens, dim = embedded_words.size()

        embedded_words = embedded_words.view(batch_size * max_num_sents, max_num_tokens, dim)

        # we encode each sentence with a seq2seq encoder on its words, then seq2vec encoder incorporating attention
        # embedded_words = self._pre_sentence_encoder_dropout(embedded_words)
        encoded_words = self._sentence_encoder(embedded_words, mask.view(batch_size*max_num_sents, -1))#.unsqueeze(1))
        sentence_repr = self._word_attention(encoded_words, mask)
        sentence_repr = torch.sum(sentence_repr, 1)
        sentence_repr = sentence_repr.view(batch_size, max_num_sents, -1)

        # we encode each document with a seq2seq encoder on its sentences, then seq2vec encoder incorporating attention
        # sentence_repr = self._pre_document_encoder_dropout(sentence_repr)
        encoded_sents = self._document_encoder(sentence_repr, sentence_level_mask)
        document_repr = self._sentence_attention(encoded_sents, sentence_level_mask)
        document_repr = torch.sum(document_repr, 1)
        return document_repr
    
    @overrides
    def get_input_dim(self) -> int:
        return self.input_dim

    @overrides
    def get_output_dim(self) -> int:
        return self.output_dim

    
"""
class HanEncoder(Seq2VecEncoder):
    def __init__(self,
                 input_dim : int = None,
                 context_vector_dim: int = None) -> None:
        super().__init__()
        self.han = SimpleHanAttention(input_dim, context_vector_dim)

    @overrides
    def get_input_dim(self) -> int:
        return self.han.get_input_dim()

    @overrides
    def get_output_dim(self) -> int:
        return self.han.get_output_dim()
    
    @overrides
    def forward(self, tokens: torch.Tensor, mask: torch.Tensor):
        with torchsnooper.snoop():
            attended = self.han(tokens, mask)
            return attended.sum(dim=-2)


@Seq2VecEncoder.register("han_attention")
class HanAttention(Seq2VecEncoder):
    def __init__(self,
                 input_dim: int = None,
                 context_vector_dim: int = None) -> None:
        super(HanAttention, self).__init__()
        self._mlp = torch.nn.Linear(input_dim, context_vector_dim, bias=True)
        self._context_dot_product = torch.nn.Linear(context_vector_dim, 1, bias=False)
        self.vec_dim = self._mlp.weight.size(1)

    def get_input_dim(self) -> int:
        return self.vec_dim

    def get_output_dim(self) -> int:
        return self.vec_dim

    def forward(self, tokens: torch.Tensor, mask: torch.Tensor):  # pylint: disable=arguments-differ
        assert mask is not None
        batch_size, sequence_length, embedding_dim = tokens.size()

        attn_weights = tokens.view(batch_size * sequence_length, embedding_dim)
        attn_weights = torch.tanh(self._mlp(attn_weights))
        attn_weights = self._context_dot_product(attn_weights)
        attn_weights = attn_weights.view(batch_size, -1)  # batch_size x seq_len
        attn_weights = masked_softmax(attn_weights, mask)
        attn_weights = attn_weights.unsqueeze(2).expand(batch_size, sequence_length, embedding_dim)

        return torch.sum(tokens * attn_weights, 1)
"""