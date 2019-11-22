local input_units = import 'remembert/index.jsonnet';

local EMBEDDING_DIM = 300;

local MULTIHEAD_ENCODER = {
    "type": "multi_head_self_attention",
    "num_heads": 1,
    "input_dim": EMBEDDING_DIM,
    "attention_dim": EMBEDDING_DIM,
    "values_dim": EMBEDDING_DIM,
    "output_projection_dim": EMBEDDING_DIM
};

local LSTM_ENCODER = {
    "type": "lstm",
    "input_size": EMBEDDING_DIM,
    "hidden_size": EMBEDDING_DIM,
    "num_layers": 1
};

/*
local KB_INPUT_UNIT = {
    "type": "kb_input_unit",
    "pooler": BERT_POOLER,
    "kb_path": "datasets/counterspeech/kb.pt",
    "projection_dim": EMBEDDING_DIM,
    "trainable_kb": true,
    // "kb_shape": [EMBEDDING_DIM, 50],
    "encoder": LSTM_ENCODER
};
*/

{
    "model": {
        "type": "mac_network",

        "max_step": 6,
        "n_memories": 2,
        
        "self_attention": true,
        "memory_gate": true,
        "dropout": 0.1, //05
    } + input_units['cnn'],
}