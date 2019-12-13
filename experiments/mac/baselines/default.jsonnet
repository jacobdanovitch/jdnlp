local common = import '../../../experiments/common/index.jsonnet'; // from run/root directory (jdnlp)
local models = import 'models.jsonnet';

local BERT_READER = (import '../default.jsonnet').bert_reader;

local BASE_READER(use_chars=true) = {
    "type": "multiturn_reader",
    "token_indexers": {
        "tokens": {
            "type": "single_id",
            "lowercase_tokens": true
            },
        [if use_chars then "token_characters"]: {
            "type": "characters",
            "min_padding_length": 1
        }
    },
    "max_sequence_length": 1000
};

local READERS = {
    'single': BASE_READER(false) + { "concat": true, "max_turns": null },
    'multi': BASE_READER(false) + {"concat": true, "max_turns": 1 }
};

{
    config(reader, data_paths, model, positive_label, loss_weights=null):: ({
        "dataset_reader": READERS[reader],
        "model": models.config(model, positive_label=positive_label) + (if loss_weights != null then { "loss_weights": loss_weights } else {}),
        "iterator": common['iterators'].base_iterator(batch_size=16),
        "trainer": common.trainer('adam', 3e-3, validation_metric='+categorical_accuracy', patience=5)
    } + data_paths),
    bert_baseline: {
        "dataset_reader": BERT_READER(),
        "model": models.bert_config,
        "iterator": common['iterators'].base_iterator(batch_size=16),
        "trainer": common.trainer('adam', 3e-3, patience=5)
    },
}

