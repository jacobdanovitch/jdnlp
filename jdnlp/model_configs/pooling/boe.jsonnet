local EMBEDDING_DIM = 300;

{
    "model": {
        "type": "classifier_with_metrics",
        "metrics": {
            "categorical_accuracy": {},
            "f1": {
                "positive_label": 1
            }
        },
        "text_field_embedder": {
            "token_embedders": {
                "tokens": {
                    "type": "embedding",
                    "embedding_dim": EMBEDDING_DIM
                }
            }
        },
        "seq2vec_encoder": {
            "type": "boe",
            "embedding_dim": EMBEDDING_DIM
        }
    }
}
