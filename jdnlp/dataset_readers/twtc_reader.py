import json
import os

from typing import Dict, List
from overrides import overrides

from allennlp.common.file_utils import cached_path
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import LabelField, TextField, ListField
from allennlp.data.instance import Instance
from allennlp.data.tokenizers import Tokenizer, WordTokenizer
from allennlp.data.tokenizers.sentence_splitter import SpacySentenceSplitter
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer

import pandas as pd
import numpy as np

import logging
logger = logging.getLogger(__name__)


@DatasetReader.register("twtc_reader")
class TWTCDatasetReader(DatasetReader):
    """
    Reads a JSON file from the TWTC dataset.
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
        
        self.cache_data(os.path.expanduser('~/.allennlp/cache/datasets'))


    @overrides
    def _read(self, file_path):
        file_path = cached_path(file_path)
        data = pd.read_json(file_path, lines=True, orient='records')[['text', 'label']].values
        for text, label in data:
            assert isinstance(label, int)
            inst = self.text_to_instance(text, str(label))
            yield inst

    @overrides
    def text_to_instance(self, document: str, label: str = None) -> Instance:
        sentences: List[str] = self._sentence_splitter.split_sentences(document)
        tokenized_sents: List[List[str]] = (self._tokenizer.tokenize(sent) for sent in sentences)

        fields = {'tokens': ListField([TextField(s, self._token_indexers) for s in tokenized_sents])}
        if label:
            fields['label'] = LabelField(int(label), skip_indexing=True)
        return Instance(fields)