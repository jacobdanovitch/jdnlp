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

    
    def forward(self, document: torch.Tensor, sentence_per_document, word_per_sentence):
        batch_size, max_sentence_length, max_word_length = document.size()
        # |document| = (batch_size, max_sentence_length, max_word_length)
        # |sentence_per_document| = (batch_size)
        # |word_per_sentence| = (batch_size, max_sentence_length)

        #print("Document:", document.shape)

        # Remove sentence-padding in document by using "pack_padded_sequence.data"
        #print("Sentences per doc:", sentence_per_document)
        packed_sentences = pack(document,
                                lengths=sentence_per_document,#.tolist(),
                                batch_first=True,
                                enforce_sorted=False)
        # |packed_sentences.data| = (sum(sentence_length), max_word_length)
        #print("Packed sentences:", packed_sentences.data.shape)

        # Remove sentence-padding in word_per_sentence "pack_padded_sequence.data"
        # word_per_sentence = torch.tensor(word_per_sentence, device=self.device)
        #print("Words per sentence:", word_per_sentence)
        wps_padded = pad(list(map(torch.tensor, word_per_sentence)), batch_first=True)
        packed_words_per_sentence = pack(wps_padded,
                                         lengths=sentence_per_document,#.tolist(),
                                         batch_first=True,
                                         enforce_sorted=False)
        # |packed_words_per_sentence.data| = (sum(sentence_length))

        # Get sentence vectors
        sentence_vecs, word_weights = self.word_attention_model(document,
                                                                packed_words_per_sentence.data)
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
