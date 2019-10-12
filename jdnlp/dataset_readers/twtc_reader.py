from typing import Dict, List
import json
import logging

from overrides import overrides

from allennlp.common.file_utils import cached_path
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import LabelField, TextField, ArrayField, ListField, MetadataField
from allennlp.data.instance import Instance
from allennlp.data.tokenizers import Tokenizer, WordTokenizer
from allennlp.data.tokenizers.sentence_splitter import SpacySentenceSplitter
# from allennlp.data.tokenizers import WordTokenizer as SpacySentenceSplitter #TODO: DELETE
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


@DatasetReader.register("twtc_reader")
class TWTCDatasetReader(DatasetReader):
    """
    Reads a CSV file from the TWTC dataset.
    Expected format for each input line: {"report": "text", "label": "int"}
    The output of ``read`` is a list of ``Instance`` s with the fields:
        text: ``TextField``
        label: ``LabelField``
    Parameters
    ----------
    lazy : ``bool`` (optional, default=False)
        Passed to ``DatasetReader``.  If this is ``True``, training will start sooner, but will
        take longer per batch.  This also allows training with datasets that are too large to fit
        in memory.
    tokenizer : ``Tokenizer``, optional
        Tokenizer to use to split the title and abstrct into words or other kinds of tokens.
        Defaults to ``WordTokenizer()``.
    token_indexers : ``Dict[str, TokenIndexer]``, optional
        Indexers used to define input token representations. Defaults to ``{"tokens":
        SingleIdTokenIndexer()}``.
    """
    def __init__(self,
                 lazy: bool = False,
                 tokenizer: Tokenizer = None,
                 token_indexers: Dict[str, TokenIndexer] = None) -> None:
        super().__init__(lazy)
        self._tokenizer = tokenizer or WordTokenizer()
        self._sentence_splitter = SpacySentenceSplitter()
        self._token_indexers = token_indexers or {"tokens": SingleIdTokenIndexer()}


    @overrides
    def _read(self, file_path):
        data = iter(pd.read_csv(cached_path(file_path), header=None).values)
        for text, label in data:
            inst = self.text_to_instance(text, str(label))
            # if len(vars(inst.fields["tokens"])['tokens']) > 1:
            #    yield inst
            yield inst

    @overrides
    def text_to_instance(self, document: str, label: str) -> Instance:
        tokenized: List[str] = self._tokenizer.tokenize(document)

        # sentence_per_document: int = len(documents)
        # word_per_sentence: List[int] = list([len(self._tokenizer.tokenize(doc)) for doc in documents])
        
        sentences: List[str] = self._sentence_splitter.split_sentences(document)
        tokenized_sents: List[int] = (self._tokenizer.tokenize(sent) for sent in sentences)

        sent_fields = ListField([TextField(s, self._token_indexers) for s in tokenized_sents])

        # text_field = TextField(tokenized, self._token_indexers)
        # sentence_field = MetadataField(sentence_per_document)
        # word_field = MetadataField(word_per_sentence)
        label_field = LabelField(label)

        #fields = {'tokens': text_field, 'sentence_per_document': sentence_field, 'word_per_sentence': word_field, 'label': label_field}
        # fields = {'tokens': text_field, 'label': label_field}
        fields = {'tokens': sent_fields, 'label': label_field}
        return Instance(fields)