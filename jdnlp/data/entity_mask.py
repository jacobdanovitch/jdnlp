from typing import Dict, List, Tuple, Union
from overrides import overrides

from jdnlp.data import DataTransform
from jdnlp.utils.data import get_pd_fn_from_path
from jdnlp.utils.parallel import tqdm_parallel
from jdnlp.utils.text import multi_replace

import nltk
import pandas as pd

import logging
logger = logging.getLogger(__file__)

logging.basicConfig(level=logging.NOTSET)

@DataTransform.register("data_entity_masker")
class DataEntityMasker(DataTransform):
    def __init__(self, df: pd.DataFrame, tf_args: Dict[str, any]):
        super().__init__(df, tf_args)
    
    @overrides
    def _transform(self, 
        df: pd.DataFrame, 
        text_col: Union[str, int] = 0,
        new_col: Union[str, int] = None,
        processes: int = 8
    ) -> pd.DataFrame:
        docs = df[text_col]
        docs = tqdm_parallel(DataEntityMasker.mask_text, docs, processes)
        
        new_col = new_col or text_col
        df[new_col] = list(docs)
        return df
        

    @staticmethod
    def mask_text(txt: str):
        if not txt:
            return txt
        chunked = nltk.ne_chunk(nltk.tag.pos_tag(nltk.word_tokenize(txt)))
        subs = {" ".join(w for w, t in elt): elt.label() 
                for elt in chunked 
                if isinstance(elt, nltk.Tree)}
        return multi_replace(txt, subs)
