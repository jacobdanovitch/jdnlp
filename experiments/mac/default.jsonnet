local common = import '../../experiments/common/index.jsonnet'; // from run/root directory (jdnlp)
local model = import '../../jdnlp/model_configs/attention/remembert.jsonnet';

local BASE_READER(use_chars=true, max_turns=0) = {
    "dataset_reader": {
        "type": "multiturn_reader",
        "max_turns": max_turns,
        "token_indexers": {
            "tokens": {
                "type": "single_id",
                "lowercase_tokens": true
                },
            [if use_chars then "token_characters"]: {
                "type": "characters",
                "min_padding_length": 1
            }
        }
    }
};

local BERT_READER(max_turns=0) = {
    "type": "multiturn_reader",
    "max_turns": max_turns,
    "token_indexers": {
        "bert": {
            "type": "bert-pretrained", // "bert-basic",
            "pretrained_model": "bert-base-uncased"
        }
    },
    "concat": true
};

local BERT_CONFIG = (
    {
        "iterator": common.iterators.base_iterator(batch_size=16),
        "trainer": common.trainer('adam', lr=0.001, num_epochs=10)
    }
    + model.bert
);

// 2e-4
local ENCODER_CONFIG(encoder, lr=8e-5, batch_size=16) = (
    {
        "iterator": common['iterators'].base_iterator(batch_size=batch_size),
        "trainer": common.trainer('adam', lr, validation_metric='+weighted_f1', patience=5, num_epochs=2)
    }
    + model[encoder]
);

{
    reader::BASE_READER,
    bert_reader::BERT_READER,
    model:model,
    bert_config:BERT_CONFIG,
    encoder_config::ENCODER_CONFIG
}