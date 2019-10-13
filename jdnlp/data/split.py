from typing import Dict, List, Tuple, Union
from overrides import overrides

from jdnlp.data import DataTransform

from sklearn.model_selection import train_test_split
import pandas as pd

import logging
logger = logging.getLogger(__file__)

logging.basicConfig(level=logging.INFO)

@DataTransform.register("data_splitter")
class DataSplitter(DataTransform):
    def __init__(self, df: pd.DataFrame, tf_args: Dict[str, any]):
        super().__init__(df, tf_args)
    
    @overrides
    def _transform(self, df, **kwargs):
        return DataSplitter.data_split(df, **kwargs)
    
    @staticmethod
    def data_split(df: pd.DataFrame, test_size: int = 0.2, seed: int = 0):
        train, test = train_test_split(df, test_size=test_size, random_state=seed)
        
        logger.info(f'Train size: {len(train)}')
        logger.info(f'Test size: {len(test)}')
        
        return train, test
