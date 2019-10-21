local BERT_EMBEDDER = {
    "token_embedders": {
        "tokens": {
            "type": "bert-pretrained",
            "pretrained_model": "bert-base-uncased",
            "requires_grad": false,
            "top_layer_only": false
        }
    },
    "allow_unmatched_keys": true
};

local BERT_INPUT_UNIT = {
    "type": "input_unit",
    "pooler": {
        "type": "bert_pooler",
        "pretrained_model": "bert-base-uncased",
        "requires_grad": false
    }
};

local BASIC_EMBEDDER = {
    "token_embedders": {
        "tokens": {
            "type": "embedding",
            "embedding_dim": 768,
            "trainable": true
        }
    }
};

local LSTM_INPUT_UNIT = {
    "type": "input_unit",
    "pooler": {
        "type": "gru",
        "input_size": 768,
        "hidden_size": 768,
        "num_layers": 1
    },
    "encoder": {
        "type": "gru",
        "input_size": 768,
        "hidden_size": 768,
        "num_layers": 1
    },
};


{
    "dataset_reader": {
        "type": "multiturn_reader",
        "tokenizer": {
            "type": "word"
        },
        "use_cache": false
    },
    "train_data_path": "tests/fixtures/multiturn_sample.json",
    "validation_data_path": "tests/fixtures/multiturn_sample.json",
    "model": {
        "type": "mac_network",
        //"text_field_embedder": BASIC_EMBEDDER,
        //"input_unit": LSTM_INPUT_UNIT,
        "text_field_embedder": BERT_EMBEDDER,
        "input_unit": BERT_INPUT_UNIT
    },
    "iterator": {
        "type": "basic",
        "batch_size" : 2
    },
    "trainer": {
        "num_epochs": 1,
        "cuda_device": -1,
        "optimizer": {
            "type": "adam",
            "lr": 0.001
        }
    }
}