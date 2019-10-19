local EMBEDDING_DIM = 300;

local GRU = {
    "type": "gru",
    "input_size": EMBEDDING_DIM,
    "hidden_size": EMBEDDING_DIM
};

local LSTM = {
    "type": "lstm",
    "input_size": EMBEDDING_DIM,
    "hidden_size": EMBEDDING_DIM
};

local INTRA = {
    "type": "intra_sentence_attention",
    "input_dim": EMBEDDING_DIM,
    "output_dim": EMBEDDING_DIM
};

local STACKED = {
    "type": "stacked_self_attention",
    "input_dim": EMBEDDING_DIM,
    "hidden_dim": EMBEDDING_DIM,
    "projection_dim": EMBEDDING_DIM,
    "feedforward_hidden_dim": EMBEDDING_DIM,
    "num_layers": 1,
    "num_attention_heads": 1
};

local FEED_FWD =  {
    "input_dim": EMBEDDING_DIM,
    "num_layers": 3,
    "hidden_dims": [
        EMBEDDING_DIM,
        EMBEDDING_DIM,
        2
    ],
    "activations": [
        "relu",
        "relu",
        "linear"
    ],
    "dropout": [
        0.05,
        0.05,
        0
    ]
};

{
    "model": {
        "type": "HAN",
        "text_field_embedder": {
            "token_embedders": {
                "tokens": {
                    "type": "embedding",
                    "embedding_dim": EMBEDDING_DIM,
                    "trainable": true
                }
            }
        },
        "encoder": {
            "type": "HierarchicalAttention",
            "word_encoder": LSTM,
            "sent_encoder": GRU
        },
        "classifier_feedforward": FEED_FWD
    }
}