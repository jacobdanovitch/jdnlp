local default = import 'default.jsonnet';
local models = import 'models.jsonnet';

local data_paths = {
    "train_data_path": "datasets/iac2/train.jsonl",
    "validation_data_path": "datasets/iac2/valid.jsonl",
    "test_data_path": "datasets/iac2/test.jsonl",
    "evaluate_on_test": true
    
};


default.config(reader=std.extVar('turns'), data_paths=data_paths, model=std.extVar('model'), positive_label="sarc")