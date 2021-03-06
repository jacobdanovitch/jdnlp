from typing import *
import json

import os
import logging

from overrides import overrides

from allennlp.common.file_utils import cached_path
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import TextField, ListField, LabelField
from allennlp.data.instance import Instance
from allennlp.data.tokenizers import Tokenizer, WordTokenizer
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer

from allennlp.data.tokenizers.token import show_token
from allennlp.data.tokenizers.sentence_splitter import SpacySentenceSplitter


import pandas as pd
import numpy as np

from jdnlp.utils.data import triplet_sample_iterator
from jdnlp.utils.parallel import tqdm_parallel

logger = logging.getLogger(__name__)


@DatasetReader.register("triplet_reader")
class TripletDatasetReader(DatasetReader):
    """
    Reads a CSV file for neural news matching.
    Expected format for each input line: {"comment": "text", "article": "int"}
    The output of ``read`` is a list of ``Instance`` s with the fields:
        anchor: ``TextField``
        positve: ``TextField``
        negative: ``TextField``
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
                 comment_index_path: str,
                 article_index_path: str, 
                 lazy: bool = False,
                 tokenizer: Tokenizer = None,
                 max_seq_len: int = 100_000,
                 sample: int = None,
                 token_indexers: Dict[str, TokenIndexer] = None) -> None:
        super().__init__(lazy)
        self._tokenizer = tokenizer or WordTokenizer()
        self._token_indexers = token_indexers or {"tokens": SingleIdTokenIndexer()}
        
        # self.sent_splitter = SpacySentenceSplitter()
        
        comment_idx = pd.read_json(comment_index_path)
        self.comment_idx = dict(comment_idx.values)
        self.idx2comment = dict(comment_idx[reversed(comment_idx.columns[:2])].values)
        
        article_idx = pd.read_json(article_index_path, lines=True)
        self.article_idx = dict(article_idx.values)
        self.idx2article = dict(article_idx[reversed(article_idx.columns[:2])].values)
        
        self.max_seq_len = max_seq_len
        self.sample = sample
        
        # self.cache_data(os.path.expanduser('~/.allennlp/cache/datasets'))


    @overrides
    def _read(self, file_path):
        df = pd.read_csv(file_path).dropna() # cached_path()
        if self.sample:
            df = df.sample(self.sample)
        
        for article_idx, comment_idx in df[['text_idx', 'comment_idx']].values:
            inst = self.text_to_instance(article_idx, comment_idx, label=str(article_idx)) # 
            if inst:
                yield inst

            

    def index_to_textfield(self, i, index):
        text = self._tokenizer.tokenize(index[i][:100_000])[:self.max_seq_len]
        return TextField(text, self._token_indexers)


    """
    TODO: Specialized tokenization for Reddit (> quotes) and Twitter (# hashtags)
    """
    @overrides
    def text_to_instance(self, article_idx, comment_idx, label=None) -> Instance: # triple: Tuple[int, int, int]
        ordered_indices = [self.idx2article, self.idx2comment]# , self.idx2comment] # anchor article, positive comment, negative comment
        triple = [article_idx, comment_idx]
        
        triple_fields = [self.index_to_textfield(t, idx) for (t, idx) in zip(triple, ordered_indices)]
        if not all(triple_fields):
            return False
        
        inst = {**dict(zip(['anchor', 'positive', 'negative'], triple_fields))}
        if label:
            inst['label'] = LabelField(str(label))
        return Instance(inst)


    def field_tokens(self, inst, field, fmt):
        tokens = vars(inst.fields[field])['tokens']
        if fmt == "str": 
            return " ".join(str(t) for t in tokens)
        elif fmt:
            return [getattr(t,fmt) for t in tokens]
        
        return [show_token(t) for t in tokens]


    def preview(self, n, fmt):
        fmt = fmt or "str"
        reader = TripletDatasetReader()
        assert False, "Update the data path on the next line"
        instances = reader.read('tests/fixtures/nnm_sample.csv')
        for inst in instances[:n]:
            inst_d = {
                'anchor': self.field_tokens(inst, "anchor", fmt),
                'positive': self.field_tokens(inst, "positive", fmt),
                'negative': self.field_tokens(inst, "negative", fmt)
            }
            
            print(json.dumps(inst_d, indent=None))



