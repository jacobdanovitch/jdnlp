local defaults = {
    'embeddings': import '../../../jdnlp/model_configs/common/embeddings.jsonnet',
    'seq2vec': import '../../../jdnlp/model_configs/common/seq2vec.jsonnet',
    'seq2seq': import '../../../jdnlp/model_configs/common/seq2seq.jsonnet',
};



{
    config(model, positive_label="1", dim=300):: {
        "type": "classifier_with_metrics",
        "metrics": {
            "categorical_accuracy": {},
            "f1": {
                "positive_label": 1 // positive_label
            },
            "weighted_f1": {},
        },
        //"text_field_embedder": defaults.embeddings.char_embedder(dim, true, "glove", char_output_dim=100),
        "text_field_embedder": defaults.embeddings.basic_embedder(dim, true, "glove"),
        "seq2seq_encoder": defaults.seq2seq.rnn(dim, 'gru') + {"bidirectional": true, "hidden_size": dim/2},
        "seq2vec_encoder": defaults.seq2vec[model](dim=dim),
    },
    bert_config: {
        "type": "bert_for_classification",
        "bert_model": "bert-base-uncased",
        "dropout": 0.0
    },
}