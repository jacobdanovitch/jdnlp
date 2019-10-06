import json
import textacy

from typing import *
from functools import partial

from textacy.augmentation import transforms
from textacy.augmentation.augmenter import Augmenter

from jdnlp.utils import tqdm_parallel
from jdnlp.data.transform import unpack_text_dataset

"""
Accepts a JSON object like {tf1: {arg1: val1, ...}, ...}
Returns a textacy Augmenter.
"""
def build_augmenter(transform_dict: Dict[str, Dict[str, any]]):
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
    

def augment_transform(docs, config_path: str, processes=8):
    transforms = json.load(open(config_path))
    augmenter = build_augmenter(transforms)

    docs = tqdm_parallel(textacy.make_spacy_doc, docs, processes)
    docs = tqdm_parallel(augmenter.apply_transforms, docs, processes)

    return docs


