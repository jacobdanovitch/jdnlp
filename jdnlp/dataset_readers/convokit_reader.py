import json
import os

from typing import Dict, List
from overrides import overrides

from allennlp.common.file_utils import cached_path
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import LabelField, TextField, ListField
from allennlp.data.instance import Instance
from allennlp.data.tokenizers import Token, Tokenizer, WordTokenizer#, SpacyTokenizer
from allennlp.data.token_indexers import TokenIndexer, PretrainedBertIndexer, SingleIdTokenIndexer

from convokit import download, Corpus

import pandas as pd
import numpy as np

from IPython.display import display
import logging
logger = logging.getLogger(__name__)

@DatasetReader.register("convokit_reader")
class ConvoKitReader(DatasetReader):
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
                 label_field: str,
                 forecast: bool = False,
                 
                 max_sequence_length: int = None, # 510,
                 max_turns: int = 3,
                 concat: bool = False,
                 
                 bert_model: str = 'bert-base-uncased',
                 
                 lazy: bool = False,
                 use_cache: bool = False,
                 
                 tokenizer: Tokenizer = WordTokenizer(),
                 token_indexers: Dict[str, TokenIndexer] = None) -> None:
        super().__init__(lazy)
        """
        self._default_indexer = PretrainedBertIndexer(pretrained_model=bert_model)
        
        self.use_default_indexer = not bool(tokenizer)
        if self.use_default_indexer:
            logger.warn(f'Using BERT indexer.')
        """
        
        self._tokenizer = tokenizer.tokenize if tokenizer else self.tokenizer
        self._token_indexers = token_indexers or {"tokens": SingleIdTokenIndexer() if tokenizer else self._default_indexer}
        
        self.max_turns = max_turns or 0
        self.max_sequence_length = max_sequence_length
        self.concat = concat
        
        self.label_field = label_field
        self.forecast = forecast
        
        self.use_cache = use_cache
        if self.use_cache:
            raise NotImplementedError("Can't cache this corpus yet (need to differentiate splits)")
            # self.cache_data(os.path.expanduser('~/.allennlp/cache/datasets'))

    def tokenizer(self, s: str):
        return self._default_indexer.wordpiece_tokenizer(s)


    @overrides
    def _read(self, corpus_split):        
        corpus_split = corpus_split.split('_')
        
        corpus_name = corpus_split[0]
        self.split = corpus_split[1] if len(corpus_split) > 1 else None
        
        # if self.use_cache: corpus_name = cached_path(corpus_name)

        corpus = Corpus(filename=download(corpus_name))
        for conv in corpus.iter_conversations():
            meta = conv.meta
            
            if self.split and (meta['split'] != self.split):
                continue
            
            label = str(meta[self.label_field])
            turns = [u.text for u in conv.iter_utterances() if u.text.strip() and (not u.meta.get('is_section_header'))]
            
            end = len(turns)-1 if self.forecast else None
            turns = turns[-self.max_turns:end]
            
            if turns:
                yield self.text_to_instance(turns, label)

    @overrides
    def text_to_instance(self, turns: str, label: str = None) -> Instance:
        def build_textfield(s):
            if self.max_sequence_length:
                s = s[-self.max_sequence_length:]
            return TextField(s, self._token_indexers)
        
        if self.concat:
            return Instance({
                'tokens': build_textfield(' '.join(turns)),
                'label': LabelField(label)
            })
        
        tokenized_turns = ([w for w in self._tokenizer(turn)] for turn in turns)
        fields = {'tokens': ListField([build_textfield(s) for s in tokenized_turns])}

        if label:
            fields['label'] = LabelField(label)
        return Instance(fields)
    
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