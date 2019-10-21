local BERT_EMBEDDER = {
    "tokens": {
        "type": "bert-pretrained",
        "requires_grad": false,
        "top_layer_only": false
    }, 
    "allow_unmatched_keys": true
};

local BERT_INPUT_UNIT = {
    "type": "bert_input_unit",
    "pooler": {
        "type": "bert_pooler",
        "pretrained_model": "bert-base-uncased",
        "requires_grad": true
    }
};

{
    "model": {
        "type": "mac_network",
        "text_field_embedder": BERT_EMBEDDER,
        "input_unit": BERT_INPUT_UNIT,
        // "dropout": 0.25
    }
}