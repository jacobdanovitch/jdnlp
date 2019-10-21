local BASIC_EMBEDDER = {
    "token_embedders": {
        "tokens": {
            "type": "embedding",
            "embedding_dim": 100,
            "trainable": true
        }
    }
};

local BASE_ITERATOR = {
    "type": "basic",
    "batch_size" : 32
};

local MULTIPROCESS_ITERATOR = {
    "type": "multiprocess", 
    "base_iterator": BASE_ITERATOR,
    "num_workers": 16
};


{
    "dataset_reader": {
        "type": "multiturn_reader",
        "use_cache": false,
    },
    "train_data_path": "datasets/dialog-safety/train_resampled.json",
    "validation_data_path": "datasets/dialog-safety/valid.json",
    // "test_data_path": "datasets/dialog-safety/test.json",

    // "train_data_path": "datasets/dialog-safety/dev.json",
    // "validation_data_path": "datasets/dialog-safety/dev.json",

    // "model.loss_weights": [0.15, 2.25],
    
    "iterator": BASE_ITERATOR,
    "trainer": {
        "num_epochs": 5,
        "cuda_device": 0,
        "optimizer": {
            "type": "bert_adam",
            "lr": 1e-5, // 5e-5,
            "t_total": -1,
            "max_grad_norm": 1.0,
            "weight_decay": 0.01,
            "parameter_groups": [
              [["bias", "LayerNorm.bias", "LayerNorm.weight", "layer_norm.weight"], {"weight_decay": 0.0}],
            ],
        },

        /*
        "learning_rate_scheduler": {
            "type": "slanted_triangular",
            "num_epochs": 5,
            "num_steps_per_epoch": 8829,
        },*/
    }
}