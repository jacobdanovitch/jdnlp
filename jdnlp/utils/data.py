import pandas as pd
import numpy as np

def get_df_from_file(fp: str, **kwargs):
    read_fn = get_pd_fn_from_path(fp, 'read')
    return read_fn(fp, **kwargs)


def get_pd_fn_from_path(fp: str, fn: str, module=pd):
    ftype = fp.split('.')[-1]
    fn_name = f'{fn}_{ftype}'

    if hasattr(module, fn_name):
        pd_fn = getattr(module, fn_name)
        return pd_fn
    
    raise AttributeError(f'Invalid method {fn_name} (from module {module} path {fp})')



def triplet_sample_iterator(df: pd.DataFrame, random_state: np.random.RandomState, max_idx: int):
    """
    Note to self: For the tweet relevance experiment, we have (tweet, article).

    Parameters
    ----------
    df : pd.DataFrame
        Two-column dataframe. The first column is the anchor data, the second is a positive example.

    Returns
    -------
    An iterator consisting of:
    triplet : Tuple
        A tuple containing (anchor, positive, negative).
    """    

   neg_sample = lambda: random_state.randint(0, max_idx+1)
   for (anchor, positive) in df.loc[:, df.columns[:3]].values.tolist():
       negative = neg_sample()
       while positive == negative:
           negative = neg_sample()
       
       yield anchor, positive, negative