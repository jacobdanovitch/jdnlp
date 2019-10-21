"""
References:
* https://elitedatascience.com/imbalanced-classes
* https://beckernick.github.io/oversampling-modeling/
"""

from typing import Dict, List, Tuple, Union
from overrides import overrides

from jdnlp.data import DataTransform
from jdnlp.utils.data import get_pd_fn_from_path

import pandas as pd
from sklearn.utils import resample

from IPython.core.display import display

@DataTransform.register("data_resampler")
class DataResampler(DataTransform):
    def __init__(self, df: pd.DataFrame, tf_args: Dict[str, any]):
        super().__init__(df, tf_args)
    
    @overrides
    def _transform(self, 
        df: pd.DataFrame, 
        minority_class: any, 
        sample_pct: float,
        sample_up: bool = True, 
        label_col: Union[str, int] = 1,
        write_config: Dict[str, any] = {},
        show_counts: bool = True,
        seed: int = 0
    ) -> pd.DataFrame:
        labels = df[label_col]
        assert (labels == minority_class).any()

        df_majority = df[labels != minority_class]
        df_minority = df[labels == minority_class]

        if sample_up:
            df_minority_upsampled = resample(df_minority, 
                                            replace=True,   
                                            n_samples=int(len(df_minority)*(1+sample_pct)),    
                                            random_state=seed) 

            df_resampled = pd.concat([df_majority, df_minority_upsampled])
            print(len(df_resampled))
        else:
            df_majority_downsampled = resample(df_majority, 
                                            replace=False,
                                            n_samples=int(len(df_majority)*(1-sample_pct)), 
                                            random_state=seed)
            df_resampled = pd.concat([df_minority, df_majority_downsampled])
        
        if write_config:
            file_path = write_config.pop('file_path')
            write_fn = get_pd_fn_from_path(file_path, 'to', module=pd.DataFrame)
            write_fn(df, **write_config)

        if show_counts:
            display('Before:')
            display(df[label_col].value_counts())

            display('\nAfter:')
            display(df_resampled[label_col].value_counts())
        
        return df_resampled
            

        
