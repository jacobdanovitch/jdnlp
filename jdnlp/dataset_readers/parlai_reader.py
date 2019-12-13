import json
import os

from typing import Dict, List
from overrides import overrides

from allennlp.common.file_utils import cached_path
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import LabelField, TextField, ListField
from allennlp.data.instance import Instance
from allennlp.data.tokenizers import Token, Tokenizer, WordTokenizer#, SpacyTokenizer
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer

from jdnlp.dataset_readers.multiturn_reader import MultiTurnReader

from nltk import ngrams
import itertools

import pandas as pd
import numpy as np

from IPython.display import display
import logging
logger = logging.getLogger(__name__)

@DatasetReader.register("parlai_reader")
class ParlAIReader(MultiTurnReader):
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
    
    def turns_to_ngrams(self, dialog):
        def turn_to_row(turn):
            return dict(text = [t['text'] for t in turn], label = turn[-1][self.label_field])
        
        turns = ngrams(dialog, self.max_turns)
        turns = map(turn_to_row, turns)
        return pd.DataFrame(turns)
    
    def process_df(self, fp):
        df = pd.read_json(fp, orient='records', lines=True)
        if self.label_field in columns:
            df['text'] = df['dialogue'].apply(lambda conv: [x['text'] for x in conv])
            return df[['text', 'label']]
        
        
        
        logger.warning(f'Using experimental ngrams mode (n={self.max_turns}).')
        
        text = df['dialogue'].apply(self.turns_to_ngrams)
        
        
        
        
        
    
    @overrides
    def _read(self, fp):        
        df = self.process_df(fp)
        for (turns, label) in df[['text', 'label']].values:
            label = str(meta[self.label_field])
            
            end = len(turns)-1 if self.forecast else None
            turns = turns[-self.max_turns:end]
            
            if turns:
                yield self.text_to_instance(turns, label)