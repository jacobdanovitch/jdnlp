from typing import *
from collections import OrderedDict

import pandas as pd
import numpy as np


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
def triplet_sample_iterator(df: pd.DataFrame, random_state: np.random.RandomState):
    a2p: Dict[any, any] = OrderedDict(df.loc[:, df.columns[:2]].values.tolist())
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
