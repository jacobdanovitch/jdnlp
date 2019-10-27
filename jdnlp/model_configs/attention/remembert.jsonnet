local BERT_POOLER = {
    "type": "bert_pooler",
    "pretrained_model": "bert-base-uncased",
    "requires_grad": true
};

local BERT_EMBEDDER = {
    "tokens": {
        "type": "bert-pretrained",
        "pretrained_model": "bert-base-uncased",
        "requires_grad": false,
        "top_layer_only": false
    }, 
    "allow_unmatched_keys": true
};

local BERT_INPUT_UNIT = {
    "type": "input_unit",
    "pooler": BERT_POOLER
};

local EMBEDDING_DIM = 768;

local BASIC_EMBEDDER = {
    "token_embedders": {
        "tokens": {
            "type": "embedding",
            // "pretrained_file": "https://allennlp.s3.amazonaws.com/datasets/glove/glove.840B.300d.txt.gz",
            "embedding_dim": EMBEDDING_DIM,
            "trainable": true
        }
    }
};

local LSTM_POOLER = {
    "type": "lstm",
    "input_size": EMBEDDING_DIM,
    "hidden_size": EMBEDDING_DIM
};

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

local KB_INPUT_UNIT = {
    "type": "kb_input_unit",
    "pooler": BERT_POOLER,
    "kb_path": "datasets/counterspeech/kb.pt",
    "projection_dim": EMBEDDING_DIM,
    "trainable_kb": true,
    // "kb_shape": [EMBEDDING_DIM, 50],
    "encoder": LSTM_ENCODER
};

{
    "model": {
        "type": "mac_network",
        "text_field_embedder": BERT_EMBEDDER, // BASIC_EMBEDDER,
        "input_unit": KB_INPUT_UNIT,
        "max_step": 6,
        "n_memories": 2,
        "self_attention": true,
        "memory_gate": true,
        "dropout": 0.05
    }
}