local BERT_EMBEDDER = {
    "tokens": {
        "type": "bert-pretrained",
        "pretrained_model": "bert-base-uncased",
        "requires_grad": false,
        "top_layer_only": false
    }, 
    "allow_unmatched_keys": true
};


local BERT_POOLER = {
    "type": "bert_pooler",
    "pretrained_model": "bert-base-uncased",
    "requires_grad": true
};

local BERT_INPUT_UNIT = {
    "type": "input_unit",
    "pooler": BERT_POOLER
};

{
    "text_field_embedder": BERT_EMBEDDER,
    "input_unit": BERT_INPUT_UNIT
}