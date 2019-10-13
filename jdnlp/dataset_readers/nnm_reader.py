from typing import *
import json
import logging

from overrides import overrides

from allennlp.common.file_utils import cached_path
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import TextField, ListField, LabelField
from allennlp.data.instance import Instance
from allennlp.data.tokenizers import Tokenizer, WordTokenizer
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer

from allennlp.data.tokenizers.token import show_token


import pandas as pd
import numpy as np

from jdnlp.utils.data import triplet_sample_iterator
from jdnlp.utils.parallel import tqdm_parallel

logger = logging.getLogger(__name__)


@DatasetReader.register("nnm_reader")
class NNMDatasetReader(DatasetReader):
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
                 yield_triples: bool = False,
                 neg_prob: float = 0.5,
                 random_seed: int = 0,
                 lazy: bool = False,
                 tokenizer: Tokenizer = None,
                 token_indexers: Dict[str, TokenIndexer] = None) -> None:
        super().__init__(lazy)
        self._tokenizer = tokenizer or WordTokenizer()
        self._token_indexers = token_indexers or {"tokens": SingleIdTokenIndexer()}
        
        self.yield_triples = yield_triples
        self.p = [neg_prob, 1-neg_prob]
        self.random_state = np.random.RandomState(random_seed)
        
        comment_idx = pd.read_json(comment_index_path)
        self.comment_idx = dict(comment_idx.values)
        self.idx2comment = dict(comment_idx[reversed(comment_idx.columns[:2])].values)
        
        article_idx = pd.read_json(article_index_path)
        self.article_idx = dict(article_idx.values)
        self.idx2article = dict(article_idx[reversed(article_idx.columns[:2])].values)


    @overrides
    def _read(self, file_path):
        df = pd.read_csv(cached_path(file_path))
        df = df.dropna()
        
        # logger.info(df[["comment_idx", "text_idx"]].max())
        c_max, t_max = df[["comment_idx", "text_idx"]].max()
        triples = triplet_sample_iterator(df, self.random_state, t_max)
        # """
        for triple in triples:
            inst = self.text_to_instance(triple)
            if inst:
                yield inst
        # """
        # triples = filter(bool, triples)
        # return filter(bool, tqdm_parallel(self.text_to_instance, triples, 16))

            

    def index_to_textfield(self, i, index):
        text = index[i]
        if len(text) > 100_000:
            return False
        return TextField(self._tokenizer.tokenize(text), self._token_indexers)


    """
    TODO: Specialized tokenization for Reddit (> quotes) and Twitter (# hashtags)
    """
    @overrides
    def text_to_instance(self, triple: Tuple[int, int, int], label=None) -> Instance:
        # logger.info(triple)
        ordered_indices = [self.idx2comment, self.idx2article, self.idx2article] # anchor comment, positive article, negative article
        # ordered_indices = [self.comment_idx, self.article_idx, self.article_idx]
        
        assert len(triple) == len(ordered_indices)
        triple_fields = [self.index_to_textfield(t, idx) for (t, idx) in zip(triple, ordered_indices)]
        if not all(triple_fields):
            return False

        anchor, positive, negative = triple_fields
        
        if self.yield_triples:    
            fields = {
                'anchor': anchor,
                'positive': positive,
                'negative': negative
            }
            return Instance(fields)

        label = self.random_state.choice([0, 1], p=self.p)
        sample = [negative, positive][label]
        fields = {
            'left': anchor,
            'right': sample,
            'label': LabelField(int(label), skip_indexing=True)
        }

        return Instance(fields)


    def field_tokens(self, inst, field, fmt):
        tokens = vars(inst.fields[field])['tokens']
        if fmt == "str": 
            return " ".join(str(t) for t in tokens)
        elif fmt:
            return [getattr(t,fmt) for t in tokens]
        
        return [show_token(t) for t in tokens]


    def preview(self, n, fmt):
        fmt = fmt or "str"
        reader = NNMDatasetReader()
        instances = reader.read('tests/fixtures/nnm_sample.csv')
        for inst in instances[:n]:
            inst_d = {
                'anchor': self.field_tokens(inst, "anchor", fmt),
                'positive': self.field_tokens(inst, "positive", fmt),
                'negative': self.field_tokens(inst, "negative", fmt)
            }
            
            print(json.dumps(inst_d, indent=None))



