from typing import *
from collections import OrderedDict

import pandas as pd
import numpy as np

import logging
logger = logging.getLogger(__name__)

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
def _triplet_sample_iterator(df: pd.DataFrame, random_state: np.random.RandomState):
    a2p: Dict[any, any] = dict(df.loc[:, df.columns[:2]].values.tolist())
    for (anchor, positive) in a2p.items():
        """
        Search for negative example. 
        Choose one anchor randomly, then ensure their labels are different, because
        anchors can share labels.
        """
        negative_anchor = random_state.choice(list(a2p.keys()))
        while a2p[negative_anchor] == positive:
            negative_anchor = random_state.choice(list(a2p.keys()))
        
        negative = a2p[negative_anchor]

        yield anchor, positive, negative


def __triplet_sample_iterator_df(df, random_state):
    df = df.dropna()
    for (anchor, positive) in df.loc[:, df.columns[:2]].values.tolist():
        negative = list(df[df[df.columns[1]] != positive].sample(1))[0]
        yield anchor, positive, negative
        
        
def ___triplet_sample_iterator(df, random_state):
    df = df.dropna()
    df['negative'] = df[df.columns[1]].sample(frac=1)
    
    cols = df.columns
    for (anchor, positive, negative) in df.loc[:, cols[:3]].values.tolist():
        while positive == negative:
            # logging.warning("positive == negative")
            negative = list(df[df[cols[1]] != positive].sample(1))[0]
            yield anchor, positive, negative #TODO: DELETE
        yield anchor, positive, negative
        
        
        
def triplet_sample_iterator(df, random_state, max_idx):
   neg_sample = lambda: random_state.randint(0, max_idx+1)
   for (anchor, positive) in df.loc[:, df.columns[:3]].values.tolist():
       negative = neg_sample()
       while positive == negative:
           negative = neg_sample()
       
       yield anchor, positive, negative
        
        
        
