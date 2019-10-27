local BASE_ITERATOR = {
    "type": "basic",
    "batch_size" : 8
};

local LEARNING_RATE = 1e-3;

local BERT_OPT = {
    "type": "bert_adam",
    "lr": 5e-5,
    "t_total": -1,
    "max_grad_norm": 1.0,
    "weight_decay": 0.01,
    "parameter_groups": [
        [["bias", "LayerNorm.bias", "LayerNorm.weight", "layer_norm.weight"], {"weight_decay": 0.0}],
    ],
};

local ADAM_OPT = {
    "type": "adam",
    "lr": LEARNING_RATE
};

local SGD_OPT = {
    "type": "sgd",
    "lr": LEARNING_RATE
};

local BASIC_READER = {
    "type": "text_classification_json",
    "skip_label_indexing": true,
};

// https://discourse.allennlp.org/t/bert-seq2seq-via-config-file-tying-bert-indexer-and-target/52
local BERT_READER = {
    "type": "text_classification_json",
    "skip_label_indexing": true,
    "tokenizer": {
        "type": "pretrained_transformer", // 
        "model_name": "bert-base-uncased", // "bert-pretrained",
        // "pretrained_model": "bert-base-uncased",
        "do_lowercase": true
    },
    "token_indexers": {
        "tokens": {
            "type": "pretrained_transformer",
            "model_name": "bert-base-uncased",
            "namespace": "tokens",
            "do_lowercase": true
        },
    },
    "max_sequence_length": 510
};



{
    "dataset_reader": BERT_READER,
    
    /*
    "train_data_path": "datasets/counterspeech/sample.json",
    "validation_data_path": "datasets/counterspeech/sample.json",
    */

    // /*
    "train_data_path": "datasets/counterspeech/train.json",
    "validation_data_path": "datasets/counterspeech/valid.json",
    "test_data_path": "datasets/counterspeech/test.json",
    // */

    "evaluate_on_test": true,

    // "model.loss_weights":  [0.35, 1.25],
    "model.loss_weights": [0.17, 0.83],
    

    "iterator": BASE_ITERATOR,
    "trainer": {
        "num_epochs": 10,
        "cuda_device": 0,
        "histogram_interval": 100,
        "optimizer": BERT_OPT // ADAM_OPT,

        /*
        "learning_rate_scheduler": {
            "type": "slanted_triangular",
            "num_epochs": 5,
            "num_steps_per_epoch": 8829,
        },*/
    }
}