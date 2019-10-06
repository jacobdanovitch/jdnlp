from typing import *

from jdnlp.data.transform import unpack_text_dataset
from jdnlp.utils.text import multi_replace
from jdnlp.utils.parallel import tqdm_parallel

import nltk

def mask_text(txt: str):
    if not txt: # or pd.isnull(txt):
        return txt
    chunked = nltk.ne_chunk(nltk.tag.pos_tag(nltk.word_tokenize(txt)))
    subs = {" ".join(w for w, t in elt): elt.label() 
            for elt in chunked 
            if isinstance(elt, nltk.Tree)}
    return multi_replace(txt, subs)


def entity_mask_transform(docs: List[str], processes=8):
    docs = tqdm_parallel(mask_text, docs, processes)
    return docs
