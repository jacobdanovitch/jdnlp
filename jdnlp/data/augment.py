from typing import Dict, List, Tuple, Union
from overrides import overrides

from functools import partial

import textacy
from textacy.augmentation import transforms
from textacy.augmentation.augmenter import Augmenter

from jdnlp.data import DataTransform
from jdnlp.utils.data import get_pd_fn_from_path
from jdnlp.utils.text import unpack_text_dataset
from jdnlp.utils.parallel import tqdm_parallel

import pandas as pd
from sklearn.utils import resample

import logging
logger = logging.getLogger(__file__)

logging.basicConfig(level=logging.NOTSET)

@DataTransform.register("data_augmenter")
class DataAugmenter(DataTransform):
    def __init__(self, df: pd.DataFrame, tf_args: Dict[str, any]):
        super().__init__(df, tf_args)
        self.augmenter = DataAugmenter.build_augmenter(self.tf_args.pop('augments'))
    
    def augment_document(self, doc):
        try:
            doc = textacy.make_spacy_doc(doc, lang="en_core_web_sm")
            doc = self.augmenter.apply_transforms(doc)
            return str(doc)
        except:
            return str(doc)
    
    @overrides
    def _transform(self, 
        df: pd.DataFrame, 
        # augments: Dict[str, Dict[str, any]],
        text_col: Union[str, int] = 0,
        processes: int = 8,
        seed: int = 0
    ) -> pd.DataFrame:
        
        original = df.copy()

        docs = df[text_col]
        docs = list(tqdm_parallel(self.augment_document, docs, processes))

        # df[text_col] = list(docs)
        df[text_col] = docs
        df = pd.concat([original, df])

        logger.info(f'Original length: {len(original)}')
        logger.info(f'New length: {len(df)}')

        print(f'Original length: {len(original)}')
        print(f'New length: {len(df)}')


        return df
            

    @staticmethod
    def build_augmenter(transform_dict: Dict[str, Dict[str, any]]):
        """
        Accepts a JSON object like {tf1: {arg1: val1, ...}, ...}
        Returns a textacy Augmenter.
        """
        
        prob = "prob" in transform_dict and transform_dict.pop("prob")
        tfs = []
        for t, kwargs in transform_dict.items():
            t = t.strip().lower()
            if not hasattr(transforms, t):
                raise AttributeError(f'Invalid transform operation specified: {t}')
            
            tf = getattr(transforms, t)
            if kwargs:
                tf = partial(tf, **kwargs)
            tfs.append(tf)
        return Augmenter(tfs, num=prob)
        

        
