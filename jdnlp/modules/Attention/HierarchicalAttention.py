from typing import Dict, Optional
from overrides import overrides

import torch
import torch.nn as nn

from torch.nn.utils.rnn import pad_sequence as pad
from torch.nn.utils.rnn import pack_padded_sequence as pack
from torch.nn.utils.rnn import pad_packed_sequence as unpack
from torch.nn.utils.rnn import PackedSequence

from allennlp.modules import Seq2VecEncoder

from jdnlp.modules.Attention.WordAttention import WordAttention
from jdnlp.modules.Attention.SentenceAttention import SentenceAttention

import logging
logger = logging.getLogger(__name__)

@Seq2VecEncoder.register("HierarchicalAttention")
class HierarchialAttention(Seq2VecEncoder):
    def __init__(self, input_size: int, hidden_size: int, attention_size: int,
                 n_layers: int = 1, dropout_p: float = 0.05, device="cpu"):
        super().__init__()

        self.word_attention_model = WordAttention(input_size=input_size,
                                                  hidden_size=hidden_size,
                                                  attention_size=attention_size,
                                                  n_layers=n_layers,
                                                  dropout_p=dropout_p,
                                                  device=device
                                                  ).to(device)

        self.sentence_attention_model = SentenceAttention(input_size=hidden_size,
                                                          hidden_size=hidden_size,
                                                          attention_size=attention_size,
                                                          n_layers=n_layers,
                                                          dropout_p=dropout_p,
                                                          device=device
                                                          ).to(device)

        self.device = device

    @overrides
    def get_input_dim(self) -> int:
        return self.word_attention_model.input_size

    @overrides
    def get_output_dim(self) -> int:
        return self.sentence_attention_model.hidden_size


    @overrides
    def forward(self, document: torch.Tensor, sentence_per_document, word_per_sentence):
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

        # Get sentence vectors
        sentence_vecs, word_weights = self.word_attention_model(document)
        # |sentence_vecs| = (sum(sentence_length), hidden_size)
        # |word_weights| = (sum(sentence_length, max(word_per_sentence))
        # print("Sentence vecs:", sentence_vecs.size())

        # "packed_sentences" have same information to recover PackedSequence for sentence
        packed_sentence_vecs = PackedSequence(data=sentence_vecs,
                                              batch_sizes=packed_sentences.batch_sizes,
                                              sorted_indices=packed_sentences.sorted_indices,
                                              unsorted_indices=packed_sentences.unsorted_indices)

        # print("Packed sentences:", packed_sentence_vecs.data.size())
        # Get document vectors
        doc_vecs, sentence_weights = self.sentence_attention_model(sentence_vecs, sentence_per_document)
        # doc_vecs, sentence_weights = self.sentence_attention_model(sentence_vecs, sentence_per_document)
        # |doc_vecs| = (batch_size, hidden_size)
        # |sentence_weights| = (batch_size)

        return doc_vecs, sentence_weights, word_weights
