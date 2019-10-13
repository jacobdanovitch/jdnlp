import logging
import os
from typing import Dict, Union, List, Set, Tuple, Callable

import numpy as np
import pandas as pd

from allennlp.common.checks import ConfigurationError
from allennlp.common.params import Params
from allennlp.common.registrable import Registrable

from jdnlp.utils.text import unpack_text_dataset
from jdnlp.utils.data import get_df_from_file
from jdnlp.utils.cli import from_params_with_check

logger = logging.getLogger(__name__)


class DataTransform(Registrable):
    def __init__(self, df: pd.DataFrame, tf_args: Dict[str, any]):
        super().__init__()
        
        self.df = df.copy()
        self.tf_args = tf_args
    

    def transform(self):
        return self._transform(self.df, **self.tf_args)
    
    def _transform(self, data: List[Tuple[str, any]]):
        raise NotImplementedError()

    def apply_transforms(self, data: List[Tuple[str, any]], transforms: List[Callable]):
        docs, *y = unpack_text_dataset(data)
        if len(y) == 1:
            y = y[0]
        
        transformed_docs = list(docs)
        extra_ys = list(y)
        for fn in transforms:
            transformed_docs.extend(fn(docs))
            extra_ys.extend(y)
        
        return transformed_docs, extra_ys
    

    @classmethod
    def from_params(cls, params: Params, **extras) -> 'DataTransform':
        transform_type = from_params_with_check(params, "type", message_override="transform type")
        klass = DataTransform.by_name(transform_type)

        read_cfg = from_params_with_check(params, "data")
        fp = read_cfg.pop('path')
        
        df = get_df_from_file(fp, **read_cfg)

        args = from_params_with_check(params, "transform")
        tf = klass(df, args)

        return tf
        
