local default = import 'default.jsonnet';
local common = import '../../experiments/common/index.jsonnet'; // from run/root directory (jdnlp)

local data_paths = {
    "train_data_path": "datasets/iac2/train.jsonl",
    "validation_data_path": "datasets/iac2/valid.jsonl",
    // "test_data_path": "datasets/iac2/test.jsonl",
    // "evaluate_on_test": true
};

local READER(use_chars=true) = default.reader(use_chars) + data_paths;

default.encoder_config(std.extVar('model'), lr=8e-5, batch_size=16) + READER(false)