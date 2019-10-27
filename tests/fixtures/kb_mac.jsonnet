local BASE_ITERATOR = {
    "type": "basic",
    "batch_size" : 2
};

local LEARNING_RATE = 1e-3;

local ADAM_OPT = {
    "type": "adam",
    "lr": LEARNING_RATE
};


local EMBEDDING_DIM = 300;

local BASIC_EMBEDDER = {
    "token_embedders": {
        "tokens": {
            "type": "embedding",
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

local LSTM_ENCODER = {
    "type": "lstm",
    "input_size": EMBEDDING_DIM,
    "hidden_size": EMBEDDING_DIM,
    "num_layers": 1
};

local KB_INPUT_UNIT = {
    "type": "kb_input_unit",
    "pooler": LSTM_POOLER,
    "kb_path": "datasets/counterspeech/kb.pt",
    "projection_dim": EMBEDDING_DIM,
    "trainable_kb": true,
    // "kb_shape": [EMBEDDING_DIM, 50],
    // "encoder": LSTM_ENCODER
};

{
    "dataset_reader": {
        "type": "text_classification_json",
        "skip_label_indexing": true
    },
    
    "train_data_path": "datasets/counterspeech/sample.json",
    "validation_data_path": "datasets/counterspeech/sample.json",

    "model": {
        "type": "mac_network",
        "text_field_embedder": BASIC_EMBEDDER,
        "input_unit": KB_INPUT_UNIT,
        "loss_weights": [0.17, 0.83],
    },
    

    "iterator": BASE_ITERATOR,
    "trainer": {
        "num_epochs": 1,
        "cuda_device": -1,
        "optimizer": ADAM_OPT
    }
}