from jdnlp.utils.data import get_pd_fn_from_path

import json
import os
import re

from typing import Dict, List
from overrides import overrides

from allennlp.common.file_utils import cached_path
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import LabelField, TextField, ListField
from allennlp.data.instance import Instance
from allennlp.data.tokenizers import Token, Tokenizer, WordTokenizer#, SpacyTokenizer
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer

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
                 label_field: str = "label",
                 forecast: bool = False,
                 
                 max_sequence_length: int = None, # 510,
                 max_turns: int = 3,
                 concat: bool = False,
                 metadata_fields: List = [],
                 
                 lazy: bool = False,
                 sample: int = None,
                 
                 tokenizer: Tokenizer = WordTokenizer(),
                 token_indexers: Dict[str, TokenIndexer] = None) -> None:
        super().__init__(lazy)        
        self._tokenizer = tokenizer.tokenize # if tokenizer else self.tokenizer
        self._token_indexers = token_indexers or {"tokens": SingleIdTokenIndexer() if tokenizer else self._default_indexer}
        
        self.max_turns = max_turns or 0
        self.max_sequence_length = max_sequence_length
        self.concat = concat
        
        self.label_field = label_field
        self.metadata_fields = metadata_fields
        
        self.forecast = forecast
        self.sample = sample

    def tokenizer(self, s: str):
        return self._default_indexer.wordpiece_tokenizer(s)


    @overrides
    def _read(self, file_path):
        data = load_df(file_path)
        data = data[['text', self.label_field, *self.metadata_fields]].dropna(subset=['text']).values
        for turns, label, *metadata in data:
            end = len(turns)-1 if self.forecast else None
            turns = turns[-self.max_turns:end]
            if not all(turns):
                continue
            inst = self.text_to_instance(turns, str(label), *metadata)
            if inst:
                yield inst

    @overrides
    def text_to_instance(self, turns: str, label: str = None, *metadata) -> Instance:
        def build_textfield(s):
            if self.max_sequence_length:
                s = s[-self.max_sequence_length:]
            return TextField(s, self._token_indexers)
        
        if self.concat:
            text = ' '.join(turns)
            tokens = self._tokenizer(text)
            return Instance({
                'tokens': build_textfield(tokens),
                'label': LabelField(label)
            })
        
        tokenized_turns = [self._tokenizer(turn) for turn in turns]
        if not all(map(lambda x: len(x) > 3, tokenized_turns)):
            return None
        fields = {'tokens': ListField([build_textfield(s) for s in tokenized_turns])}

        if label:
            fields['label'] = LabelField(label)
            
        for field, val in zip(self.metadata_fields, metadata):
            fields[field] = LabelField(val, label_namespace=field)
        return Instance(fields)
    
def load_df(file_path):
    if re.search(r'\.json[l]*$', file_path, flags=re.M):
            data = pd.read_json(file_path, lines=True, orient='records')
    else:
        import ast
        def avoid_stupid_ast_error(row):
            try:
                return ast.literal_eval(row)
            except:
                return None
        data = pd.read_csv(file_path, converters={'text': avoid_stupid_ast_error})
    return data