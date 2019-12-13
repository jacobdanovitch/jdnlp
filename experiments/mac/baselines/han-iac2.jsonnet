local default = import '../default.jsonnet';
local common = import '../../../experiments/common/index.jsonnet'; 
local han = import '../../../jdnlp/model_configs/attention/han.jsonnet';

local data_paths = {
    "train_data_path": "datasets/iac2/train.jsonl",
    "validation_data_path": "datasets/iac2/valid.jsonl",
    "test_data_path": "datasets/iac2/test.jsonl",
    "evaluate_on_test": true
};

local READER(use_chars=true) = default.reader(use_chars) + data_paths;

{
    "model": han.model(std.extVar('model')),
    "iterator": common['iterators'].base_iterator(batch_size=8),
    "trainer": common.trainer('adam', 3e-3, validation_metric='+f1', patience=5)
} + READER(false)