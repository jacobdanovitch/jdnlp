import json
import os

from typing import Dict, List
from overrides import overrides

from allennlp.common.file_utils import cached_path
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import LabelField, TextField, ListField
from allennlp.data.instance import Instance
from allennlp.data.tokenizers import Token, Tokenizer, WordTokenizer
from allennlp.data.token_indexers import TokenIndexer, PretrainedBertIndexer, SingleIdTokenIndexer

import pandas as pd
import numpy as np

import logging
logger = logging.getLogger(__name__)

@DatasetReader.register("multiturn_reader")
class MultiTurnReader(DatasetReader):
    """
    Reads a JSON file for multi-turn conversations. Each turn should be deliminated with a new line.
    Expected format for each input line: {"text": "str", "label": "int"}
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
                 max_turns: int = 3,
                 bert_model: str = 'bert-base-uncased',
                 lazy: bool = False,
                 use_cache: bool = True,
                 tokenizer: Tokenizer = None,
                 token_indexers: Dict[str, TokenIndexer] = None) -> None:
        super().__init__(lazy)
        self._default_indexer = PretrainedBertIndexer(pretrained_model=bert_model)
        
        self.use_default_indexer = not bool(tokenizer)
        if self.use_default_indexer:
            logger.warn(f'Using BERT indexer.')
        
        self._tokenizer = tokenizer.tokenize if tokenizer else self.tokenizer
        self._token_indexers = token_indexers or {"tokens": SingleIdTokenIndexer() if tokenizer else self._default_indexer}
        
        self.max_turns = max_turns
        self.use_cache = use_cache
        if self.use_cache:
            self.cache_data(os.path.expanduser('~/.allennlp/cache/datasets'))

    def tokenizer(self, s: str):
        logger.warn("here")
        print("here")
        return self._default_indexer.wordpiece_tokenizer(s)


    @overrides
    def _read(self, file_path):
        if self.use_cache:
            file_path = cached_path(file_path)
        data = pd.read_json(file_path, lines=True, orient='records')[['text', 'label']].values
        for text, label in data:
            yield self.text_to_instance(text, str(label))

    @overrides
    def text_to_instance(self, conversation: str, label: str) -> Instance:
        turns: List[str] = conversation.split('\n')[:self.max_turns]
        tokenized_turns: List[List[str]] = ([w for w in self._tokenizer(turn)] for turn in turns)
        fields = {'tokens': ListField([TextField(s, self._token_indexers) for s in tokenized_turns])}
        # logger.warn('HERE')
        if label:
            fields['label'] = LabelField(label)
        return Instance(fields)