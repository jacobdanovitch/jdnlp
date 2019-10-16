"""
Originally implemented by: 
https://github.com/mohammadKhalifa/attention-augmented-cnn-text/blob/master/my_library/encoders/attention_cnn.py

Doesn't work yet; needs fixed-length input.
"""

from typing import Optional, Tuple

from overrides import overrides
import torch
from torch.nn import Conv1d, Linear

from allennlp.modules.seq2vec_encoders.seq2vec_encoder import Seq2VecEncoder
from allennlp.nn import Activation

import torch
import torch.nn as nn
import numpy as np
import copy
import math
import torch.nn.functional as F
from torch.autograd import Variable


import logging

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name

@Seq2VecEncoder.register("attention_cnn")
class AttnCnnEncoder(Seq2VecEncoder):
    """
    A ``AttnCnnEncoder`` is a self-attention based multiple convolution layers and max pooling layers.  As a
    :class:`Seq2VecEncoder`, the input to this module is of shape ``(batch_size, num_tokens,
    input_dim)``, and the output is of shape ``(batch_size, output_dim)``.

    The CNN has one convolution layer for each ngram filter size. Each convolution operation gives
    out a vector of size num_filters. The number of times a convolution layer will be used
    is ``num_tokens - ngram_size + 1``. The corresponding maxpooling layer aggregates all these
    outputs from the convolution layer and outputs the max.

    This operation is repeated for every ngram size passed, and consequently the dimensionality of
    the output after maxpooling is ``len(ngram_filter_sizes) * num_filters``.  This then gets
    (optionally) projected down to a lower dimensional output, specified by ``output_dim``.

    We then use a fully connected layer to project in back to the desired output_dim.  For more
    details, refer to "A Sensitivity Analysis of (and Practitioners’ Guide to) Convolutional Neural
    Networks for Sentence Classification", Zhang and Wallace 2016, particularly Figure 1.

    Parameters
    ----------
    embedding_dim : ``int``
        This is the input dimension to the encoder.  We need this because we can't do shape
        inference in pytorch, and we need to know what size filters to construct in the CNN.
    num_filters: ``int``
        This is the output dim for each convolutional layer, which is the number of "filters"
        learned by that layer.
    ngram_filter_sizes: ``Tuple[int]``, optional (default=``(2, 3, 4, 5)``)
        This specifies both the number of convolutional layers we will create and their sizes.  The
        default of ``(2, 3, 4, 5)`` will have four convolutional layers, corresponding to encoding
        ngrams of size 2 to 5 with some number of filters.
    conv_layer_activation: ``Activation``, optional (default=``torch.nn.ReLU``)
        Activation to use after the convolution layers.
    output_dim : ``Optional[int]``, optional (default=``None``)
        After doing convolutions and pooling, we'll project the collected features into a vector of
        this size.  If this value is ``None``, we will just return the result of the max pooling,
        giving an output of shape ``len(ngram_filter_sizes) * num_filters``.
    """
    def __init__(self,
                 embedding_dim: int,
                 num_filters: int,
                 ngram_filter_sizes: Tuple[int, ...] = (3, 4, 5),  # pylint: disable=bad-whitespace
                 conv_layer_activation: Activation = None,
                 output_dim: Optional[int] = None,
                 use_self_attention: Optional[bool] = True,
                 n_attention_heads: Optional[int] = 2,
                 sequence_length : Optional[int] = 30) -> None:
        super(AttnCnnEncoder, self).__init__()
        
        self._use_self_attention = use_self_attention
        logger.info("Using Self Attention!")
        if self._use_self_attention:
            assert embedding_dim % n_attention_heads == 0, "Embeddings dimension =%d" \
            "is not divisible by the number of attention heads=%d" %(embedding_dim, n_attention_heads)
        
        self._embedding_dim = embedding_dim
        self._num_filters = num_filters
        self._ngram_filter_sizes = ngram_filter_sizes
        self._activation = conv_layer_activation or Activation.by_name('relu')()
        self._output_dim = output_dim
        self._n_attention_heads = n_attention_heads
        self._positional_encodings = PositionalEncoding(self._embedding_dim)
        self._sequence_length= sequence_length

        self._multihead_attn = MultiHeadedAttention(self._n_attention_heads, self._embedding_dim)
        self._convolution_layers = [Conv1d(in_channels=self._embedding_dim,
                                           out_channels=self._num_filters,
                                           kernel_size=ngram_size)
                                    for ngram_size in self._ngram_filter_sizes]
                                
        for i, conv_layer in enumerate(self._convolution_layers):
            self.add_module('conv_layer_%d' % i, conv_layer)

        maxpool_output_dim = self._num_filters * len(self._ngram_filter_sizes)
        
        logger.warn("sequence length = %d " %(self._sequence_length))
        logger.warn("embeddings dim = %d" %(self._embedding_dim))

        if self._output_dim:
            if self._use_self_attention:
                self.projection_layer = Linear(maxpool_output_dim + (self._embedding_dim* self._sequence_length), self._output_dim)
            else :
                self.projection_layer = Linear(maxpool_output_dim, self._output_dim)
        else:
            self.projection_layer = None
            self._output_dim = maxpool_output_dim
        logger.warn('output dim = %d' % (self._output_dim))
        logger.info(self.projection_layer)
        assert self._use_self_attention

    @overrides
    def get_input_dim(self) -> int:
        return self._embedding_dim

    @overrides
    def get_output_dim(self) -> int:
        return self._output_dim

    def forward(self, tokens: torch.Tensor, mask: torch.Tensor):  # pylint: disable=arguments-differ
        if mask is not None:
            tokens = tokens * mask.unsqueeze(-1).float()
        # Our input is expected to have shape `(batch_size, num_tokens, embedding_dim)`.  The
        # convolution layers expect input of shape `(batch_size, in_channels, sequence_length)`,
        # where the conv layer `in_channels` is our `embedding_dim`.  We thus need to transpose the
        # tensor first.
        tokens = torch.transpose(tokens, 1, 2)
        tokens_t = torch.transpose(tokens, 2, 1)
        # Each convolution layer returns output of size `(batch_size, num_filters, pool_length)`,
        # where `pool_length = num_tokens - ngram_size + 1`.  We then do an activation function,
        # then do max pooling over each filter for the whole input sequence.  Because our max
        # pooling is simple, we just use `torch.max`.  The resultant tensor of has shape
        # `(batch_size, num_conv_layers * num_filters)`, which then gets projected using the
        # projection layer, if requested.

        filter_outputs = []
        for i in range(len(self._convolution_layers)):
            convolution_layer = getattr(self, 'conv_layer_{}'.format(i))
            conv_out = self._activation(convolution_layer(tokens))
            filter_outputs.append(
                    self._activation(convolution_layer(tokens)).max(dim=2)[0]
            )

        # Now we have a list of `num_conv_layers` tensors of shape `(batch_size, num_filters)`.
        # Concatenating them gives us a tensor of shape `(batch_size, num_filters * num_conv_layers)`.
        maxpool_output = torch.cat(filter_outputs, dim=1) if len(filter_outputs) > 1 else filter_outputs[0]
        logger.info('MAXPOOL: {}'.format(maxpool_output.size()))
        
        # return tokens to their original shape for self-attention
        if self._use_self_attention:
            tokens = torch.transpose(tokens, 1, 2)
            tokens = self._positional_encodings(tokens)
            logger.info('TOKENS: {}'.format(tokens.size()))
            attn_out = self._multihead_attn(tokens, tokens, tokens).view(tokens.size(0), -1) # B x H
            logger.info('ATTN: {}'.format(attn_out.size()))

            layer_output = torch.cat([maxpool_output, attn_out], dim=-1)
            logger.info('LAYER OUTPUT: {}'.format(layer_output.size()))
        else :
            layer_output = maxpool_output

        if self.projection_layer:
            # print('Layer output: {}'.format(layer_output.size()))
            result = self.projection_layer(layer_output)
        else:
            result = layer_output
        return result
    
    
    


def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

def attention(query, key, value, mask=None, dropout=None):
    "Compute 'Scaled Dot Product Attention'"
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) \
             / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = F.softmax(scores, dim = -1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn

class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        "Take in model size and number of heads."
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)
        
    def forward(self, query, key, value, mask=None):
        "Implements Figure 2"
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)
        
        query = self.linears[0](query)
        key = self.linears[1](key)
        value = self.linears[2](value)

        # extracting heads
        query = query.view(nbatches, -1, self.h, self.d_k).transpose(2,1)
        key = key.view(nbatches, -1, self.h, self.d_k).transpose(2,1)
        value = value.view(nbatches, -1, self.h, self.d_k).transpose(2,1)


        attn_output, self.attn = attention(query, key, value, mask=mask)
        #concatenating heads output
        return self.linears[-1](attn_output.view(nbatches, -1, self.h * self.d_k))


class PositionalEncoding(nn.Module):
    "Implement the PE function."
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0., max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0., d_model, 2) *
                             -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        x = x + Variable(self.pe[:, :x.size(1)], 
                         requires_grad=False)
        return self.dropout(x)


if __name__=='__main__':

      #creating sample tokens tensor of shape (1, 5, 100)
    tokens = torch.FloatTensor(np.random.randint(1, 100, size=(5,100)))
    tokens = tokens.unsqueeze(0)
    
    PE = PositionalEncoding(100)
    tokens = PE(tokens)
    attn = MultiHeadedAttention(4, 100)

    attn_result = attn(tokens, tokens, tokens)
    