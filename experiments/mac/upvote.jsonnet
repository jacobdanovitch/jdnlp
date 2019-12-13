local default = import 'default.jsonnet';
local common = import '../../experiments/common/index.jsonnet'; // from run/root directory (jdnlp)

local data_paths = {
    "train_data_path": "datasets/upvote/train-sample.jsonl",
    "validation_data_path": "datasets/upvote/dev-sample.jsonl",
    "test_data_path": "datasets/upvote/test-sample.jsonl",
    "evaluate_on_test": true
};

local READER(use_chars=true) = default.reader(use_chars) + data_paths;

default.encoder_config(std.extVar('model'), batch_size=8) + READER(false)