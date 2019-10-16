from typing import List, Tuple

import re
import pandas as pd

# https://gist.github.com/carlsmith/b2e6ba538ca6f58689b4c18f46fef11c
def multi_replace(string, substitutions):
    if not substitutions:
        return string
    
    substrings = sorted(substitutions, key=len, reverse=True)
    regex = re.compile('|'.join(map(re.escape, substrings)))
    
    return regex.sub(lambda match: substitutions[match.group(0)], string)


def unpack_text_dataset(data: List[Tuple[str, any]]):
    if data is None:
        raise ValueError("You have passed an empty dataset to a transformation function.")
    
    docs, ys = data, None
    if isinstance(docs[0], tuple) or isinstance(docs[0], list):
        docs, *ys = zip(*data)
        #if len(ys) == 1:
        #    ys = ys[0]
    
    return docs, ys


def csv_to_jsonlines(fname):
    pd.read_csv(fname, names=['text', 'label']).to_json(fname.replace(".csv", ".json"), orient="records", lines=True)