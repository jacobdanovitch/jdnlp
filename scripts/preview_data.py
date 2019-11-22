# from jdnlp.dataset

def field_tokens(inst, field, fmt):
        tokens = vars(inst.fields[field])['tokens']
        if fmt == "str": 
            return " ".join(str(t) for t in tokens)
        elif fmt:
            return [getattr(t,fmt) for t in tokens]
        
        return [show_token(t) for t in tokens]

from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.common.util import import_submodules
import_submodules('jdnlp')

reader = DatasetReader.by_name('convokit_reader')
train = reader('conversation_has_personal_attack', max_turns=3, forecast=False, use_cache=False, lazy=True)
# trainset = train.read('conversations-gone-awry-corpus')
df = train.preview('conversations-gone-awry-corpus_test', n=None)
df.head()