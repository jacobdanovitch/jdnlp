local BASE_ITERATOR = {
    "type": "basic",
    "batch_size" : 8
};

local ADAM_OPT = {
    "type": "adam",
    "lr": 0.001
};

// "pretrained_file": "https://s3-us-west-2.amazonaws.com/allennlp/datasets/glove/glove.840B.300d.txt.gz",

{
    "dataset_reader": {
        "type": "text_classification_json",
    },
    
    "train_data_path": "datasets/counterspeech/train.json",
    "validation_data_path": "datasets/counterspeech/test.json",
    // "test_data_path": "datasets/counterspeech/test.json",

    //"train_data_path": "datasets/counterspeech/sample.json",
    //"validation_data_path": "datasets/counterspeech/sample.json",

    "model.loss_weights": [0.35, 1.25],
    
    "iterator": BASE_ITERATOR,
    "trainer": {
        "num_epochs": 5,
        "cuda_device": 0,
        "optimizer": ADAM_OPT
    }
}