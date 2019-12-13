local embedding = import '../common/embeddings.jsonnet';
local seq2vec = import '../common/seq2vec.jsonnet';

{
    "model": {
        "type": "classifier_with_metrics",
        "metrics": {
            "categorical_accuracy": {},
            "f1": {
                "positive_label": 1
            }
        },
        "text_field_embedder": embedding.basic_embedder(300, true),//, "glove"),
        "seq2vec_encoder": seq2vec.cnn(dim=300)
    }
}
