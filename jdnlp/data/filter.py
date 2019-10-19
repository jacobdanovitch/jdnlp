from typing import Dict, List, Tuple, Union
from overrides import overrides

from jdnlp.data import DataTransform
import pandas as pd

import logging
logger = logging.getLogger(__file__)

logging.basicConfig(level=logging.NOTSET)

@DataTransform.register("data_filter")
class DataFilter(DataTransform):
    def __init__(self, df: pd.DataFrame, tf_args: Dict[str, any]):
        super().__init__(df, tf_args)
    
    @overrides
    def _transform(self, 
        df: pd.DataFrame, 
        query: str
    ) -> pd.DataFrame:
        return df.query(query)
