import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn import Parameter, Linear

import logging
logger = logging.getLogger(__name__)

def new_parameter(*size):
    out = nn.Parameter(torch.randn(*size, requires_grad=True).float())
    nn.init.xavier_normal_(out)
    return out


class Attention(nn.Module):
    """ Simple multiplicative attention"""
    def __init__(self, attention_size):
        super().__init__()
        self.attention = new_parameter(attention_size, 1)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x_in, reduction_dim=-2, return_attn_distribution=False):
        """
        return_attn_distribution: if True it will also return the original attention distribution
        this reduces the one before last dimension in x_in to a weighted sum of the last dimension
        e.g., x_in.shape == [64, 30, 100] -> output.shape == [64, 100]
        Usage: You have a sentence of shape [batch, sent_len, embedding_dim] and you want to
            represent sentence to a single vector using attention [batch, embedding_dim]
        Here we use it to aggregate the lexicon-aware representation of the sentence
        In two steps we convert [batch, sent_len, num_words_in_category, num_categories] into [batch, num_categories]
        """
        # calculate attn weights
        attn_score = torch.matmul(x_in, self.attention)
        # add one dimension at the end and get a distribution out of scores
        attn_distrib = self.softmax(attn_score.squeeze()).unsqueeze(-1)
        scored_x = x_in * attn_distrib
        if reduction_dim:
            scored_x = scored_x.sum(dim=reduction_dim)
        
        if return_attn_distribution:
            return scored_x, attn_distrib.reshape(x_in.shape[0], -1)

        return scored_x