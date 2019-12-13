// "datasets/counterspeech/kb.pt",

local seq2vec = import '../../common/seq2vec.jsonnet';

local KB_INPUT_UNIT(dim, kb_path, trainable=true) = {
    "type": "kb_input_unit",
    "pooler": seq2vec.bert_pooler,
    "kb_path": kb_path,
    "projection_dim": dim,
    "trainable_kb": trainable,
    // "kb_shape": [EMBEDDING_DIM, 50],
    "encoder": seq2vec.rnn(dim=300, encoder='lstm')
};