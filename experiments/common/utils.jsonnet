local BERT_PREPROCESSORS(model_name="bert-base-uncased") = {
    "tokenizer": {
        "type": "pretrained_transformer",
        "model_name": model_name,
        "do_lowercase": true
    },
    "token_indexers": {
        "tokens": {
            "type": "pretrained_transformer",
            "model_name": model_name,
            "namespace": "tokens",
            "do_lowercase": true
        },
    },
};

local BERT_LR_SCHEDULE = {
    "type": "slanted_triangular",
    "num_epochs": 5,
    "num_steps_per_epoch": 8829,
};

{
    "bert_preprocessors":: BERT_PREPROCESSORS,
    "bert_lr_schedule": BERT_LR_SCHEDULE
}