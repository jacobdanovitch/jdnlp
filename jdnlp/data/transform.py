from typing import *

import pandas as pd
from sklearn.model_selection import train_test_split

def unpack_text_dataset(data: List[Tuple[str, any]]):
    if data is None:
        raise ValueError("You have passed an empty dataset to a transformation function.")
    
    docs, ys = data, None
    if isinstance(docs[0], tuple) or isinstance(docs[0], list):
        docs, *ys = zip(*data)
        #if len(ys) == 1:
        #    ys = ys[0]
    
    return docs, ys
    


def data_split(df: pd.DataFrame, test_size=0.2, write_path=None, seed=0):
    train, test = train_test_split(df, test_size=test_size, random_state=seed)
    if write_path:
        train.to_csv(f'{write_path}/train.csv')
        test.to_csv(f'{write_path}/test.csv')
    
    return train, test



"""
Transforms will be applied in-order.
"""
def apply_transforms(data: List[Tuple[str, any]], transforms: List[Callable]):
    docs, *y = unpack_text_dataset(data)
    if len(y) == 1:
        y = y[0]
    
    transformed_docs = list(docs)
    extra_ys = list(y)
    for fn in transforms:
        transformed_docs.extend(fn(docs))
        extra_ys.extend(ys)
    
    return transformed_docs, extra_ys


"""
Takes a dataframe of some text data and any other columns as well as a list of transformations.
Splits into train and test, applies transforms, and optionally writes to files.
Assumes column with text comes *FIRST*.
"""
def transform_and_split(df: pd.DataFrame, transforms: List[Callable], test_size=0.2, write_path=None, seed=0):
    train, test = train_test_split(df, test_size=test_size, random_state=seed)

    docs, ys = apply_transforms(train.values.tolist(), transforms)
    train = pd.DataFrame(list(zip(docs, ys)), columns=train.columns)

    if write_path:
        train.to_csv(f'{write_path}/train.csv', names=None)
        test.to_csv(f'{write_path}/test.csv', names=None)
    
    return train, test



def transforms_from_json(p):
    tfs = json.load(open(p))