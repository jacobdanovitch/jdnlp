import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn import Parameter, Linear

import logging
logger = logging.getLogger(__name__)

def new_parameter(*size):
    out = Parameter(torch.FloatTensor(*size))
    torch.nn.init.xavier_normal_(out)
    return out


class Attention(nn.Module):
    """ Simple multiplicative attention"""
    def __init__(self, attention_size):
        super().__init__()
        self.attention = new_parameter(attention_size, 1)

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
        attn_score = torch.matmul(x_in, self.attention).squeeze()
        # add one dimension at the end and get a distribution out of scores
        attn_distrib = F.softmax(attn_score.squeeze(), dim=-1).unsqueeze(-1)
        scored_x = x_in * attn_distrib
        if reduction_dim:
            scored_x = torch.sum(scored_x, dim=reduction_dim)
        
        if return_attn_distribution:
            return scored_x, attn_distrib.reshape(x_in.shape[0], -1)

        return scored_x


class _Attention(nn.Module):

    def __init__(self, hidden_size, attention_size):
        super(Attention, self).__init__()

        self.linear = nn.Linear(hidden_size, attention_size, bias=False)
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim=-1)

        ## Context vector
        self.context_weight = nn.Parameter(torch.Tensor(attention_size, 1))
        self.context_weight.data.normal_(mean=0.0, std=0.02)

    def forward(self, h_src, mask=None):
        # |h_src| = (batch_size, length, hidden_size)
        # |mask| = (batch_size, length)
        print("hsrc:", h_src.size())
        batch_size, length, hidden_size = h_src.size()

        # Resize hidden_vectors to generate weight
        weights = h_src.view(-1, hidden_size)
        weights = self.linear(weights)
        weights = self.tanh(weights)

        weights = torch.mm(weights, self.context_weight).view(batch_size, -1)
        # |weights| = (batch_size, length)

        if mask is not None:
            # Set each weight as -inf, if the mask value equals to 1.
            # Since the softmax operation makes -inf to 0, masked weights would be set to 0 after softmax operation.
            # Thus, if the sample is shorter than other samples in mini-batch, the weight for empty time-step would be set to 0.
            weights.masked_fill_(mask, -float('inf'))

        # Modified every values to (0~1) by using softmax function
        weights = self.softmax(weights)
        # |weights| = (batch_size, length)

        context_vectors = torch.bmm(weights.unsqueeze(1), h_src)
        # |context_vector| = (batch_size, 1, hidden_size)

        context_vectors = context_vectors.squeeze(1)
        # |context_vector| = (batch_size, hidden_size)

        return context_vectors, weights


    @staticmethod
    def generate_mask(length):
            mask = []

            max_length = max(length)
            for l in length:
                if max_length - l > 0:
                    # If the length is shorter than maximum length among samples,
                    # set last few values to be 1s to remove attention weight.
                    mask += [torch.cat(
                        [torch.zeros((1, l)).byte(), torch.ones((1, (max_length - l))).byte()],
                        dim=-1)]
                else:
                    # If the length of the sample equals to maximum length among samples,
                    # set every value in mask to be 0.
                    mask += [torch.zeros((1, l)).byte()]

            mask = torch.cat(mask, dim=0).bool()