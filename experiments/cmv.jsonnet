local common = import 'experiments/common/index.jsonnet'; // from run/root directory (jdnlp)

local CONVOKIT_READER(label_field="conversation_has_personal_attack") = {
    "type": "convokit_reader",
    "label_field": label_field,
    "max_turns": 3,
    "forecast": false,
};

{
    "dataset_reader": CONVOKIT_READER("has_removed_comment"), // + common['utils'].bert_preprocessors,
    
    "train_data_path": "conversations-gone-awry-cmv-corpus_train",
    "validation_data_path": "conversations-gone-awry-cmv-corpus_val",
    
    // "test_data_path": "conversations-gone-awry-corpus_test",
    //"evaluate_on_test": true,
    
    "iterator": common['iterators'].base_iterator(batch_size=8),
    "trainer": common.trainer('adam', 4e-14)
}