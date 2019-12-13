from typing import *
from overrides import overrides

import os
import re

import json
import shelve


from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import TextField, ListField, LabelField
from allennlp.data.instance import Instance
from allennlp.data.tokenizers import Tokenizer, WordTokenizer
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer

from allennlp.data.tokenizers.token import show_token
from allennlp.data.tokenizers.sentence_splitter import SpacySentenceSplitter

from nltk.tokenize import sent_tokenize, TweetTokenizer

import pandas as pd
import numpy as np

import logging
logger = logging.getLogger(__name__)


@DatasetReader.register("tweetlink_reader")
class TweetLinkReader(DatasetReader):
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
                 lazy: bool = False,
                 tokenizer: Tokenizer = None,
                 segment_sentences: bool = False,
                 sample: int = None,
                 token_indexers: Dict[str, TokenIndexer] = None) -> None:
        super().__init__(lazy)
        self._tokenizer = tokenizer or WordTokenizer()
        self._token_indexers = token_indexers or {"tokens": SingleIdTokenIndexer()}
        self.segment_sentences = segment_sentences
        self.sent_split = sent_tokenize # SpacySentenceSplitter().split_sentences
        
        self.sample = sample


    @overrides
    def _read(self, fp):
        df = pd.read_json(fp, lines=True)[['url', 'article', 'text']]
        article_mask = df.article.str.match('\w+')
        text_mask = df.text.str.match('\w+')
        
        df = df[article_mask & text_mask]
        
        if self.sample:
            # df = df.sample(self.sample)
            df = df.sort_values(by='url').reset_index(drop=True).loc[:self.sample]
        
        for url, article, comment in df.values:
            # article = re.sub(r'\s{2,}', '', article)
            # comment = re.sub(r'\s{2,}', '', comment)
            
            # logger.warn(article)
            # logger.warn(comment)

            # if not (re.search('\w+', article) and re.search('\w+', comment)):
            #    logger.warn('Tossing input example.')
            #    continue
            inst = self.text_to_instance(article, comment, label=url)  
            if inst: 
                # logger.warn(self.field_tokens(inst, 'positive', 'str'))
                yield inst

    def to_textfield(self, text):
        def _to_textfield(txt):
            tokens = self._tokenizer.tokenize(txt)
            return TextField(tokens, self._token_indexers)
        if self.segment_sentences:
            sents = self.sent_split(text)
            sents = [_to_textfield(s) for s in sents]
            assert sents and all(sents), f"No sentences found: '{text}'"
            return ListField(sents)
        return _to_textfield(text)
        
    
    """
    TODO: Specialized tokenization for Reddit (> quotes) and Twitter (# hashtags)
    """
    @overrides
    def text_to_instance(self, article, comment=None, label=None) -> Instance:
        return Instance({
            'anchor': self.to_textfield(article),
            **({'positive': self.to_textfield(comment)} if comment else {}),
            **({'label': LabelField(label)} if label else {})
        })


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


