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

from convokit import download, Corpus
import itertools

import pandas as pd
import numpy as np

from IPython.display import display
import torchsnooper
import logging
logger = logging.getLogger(__name__)

@DatasetReader.register("convokit_reader")
class ConvoKitReader(MultiTurnReader):
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
    
    @overrides
    def _read(self, corpus_split):        
        corpus_split = corpus_split.split('_')
        
        corpus_name = corpus_split[0]
        self.split = corpus_split[1] if len(corpus_split) > 1 else None
        
        corpus = Corpus(filename=download(corpus_name))
        conversations = corpus.iter_conversations()
        if self.sample:
            conversations = itertools.islice(conversations, self.sample)
        
        for conv in conversations:
            meta = conv.meta
            
            if (meta.get('split') != self.split) and (meta.get('annotation_year',2018) != 2018):
                continue
            
            label = str(meta[self.label_field])
            # turns = [u.text for u in conv.iter_utterances() if u.text.strip() and (not u.meta.get('is_section_header'))]
            turns = [u.meta.parsed for u in conv.iter_utterances() if not u.meta.get('is_section_header')]
            
            end = len(turns)-1 if self.forecast else None
            turns = turns[-self.max_turns:end]
            
            if turns and all(turns):
                inst = self.text_to_instance(turns, label)
                if inst:
                    yield inst

    
    def preview(self, corpus, n=5):
        data = self.read(corpus)
        rows = []
        for inst in data:
            turns = [' '.join(word.text for word in turn.tokens) for turn in inst.fields['tokens'].field_list]
            turn_labels = [f'turn_{i}' for i in range(len(turns))]
            
            label = inst.fields['label'].label
            
            row = {**dict(zip(turn_labels, turns)), 'label': label}
            rows.append(row)
        return pd.DataFrame(rows)