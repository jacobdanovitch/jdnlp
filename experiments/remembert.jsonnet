local BERT_INPUT_UNIT = {
    "type": "bert_input_unit",
    "pooler": {
        "type": "bert_pooler",
        "pretrained_model": "bert-base-uncased",
        "requires_grad": false
    }
};


{
    "dataset_reader": {
        "type": "multiturn_reader",
        "use_cache": true
    },
    "train_data_path": "datasets/dialog-safety/train.json",
    "validation_data_path": "datasets/dialog-safety/valid.json",
    "test_data_path": "datasets/dialog-safety/test.json",

    "model": {
        "type": "mac_network",
        "input_unit": BERT_INPUT_UNIT
    },
    "iterator": {
        "type": "basic",
        "batch_size" : 2
    },
    "trainer": {
        "num_epochs": 5,
        "cuda_device": 0,
        "optimizer": {
            "type": "adam",
            "lr": 0.001
        }
    }
}