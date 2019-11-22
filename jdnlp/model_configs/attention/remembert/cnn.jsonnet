local embedding = import '../../common/embeddings.jsonnet';

{
    "text_field_embedder": embedding.basic_embedder(300, "glove", true),
    // "text_field_embedder": embedding.basic_embedder(300, null, true),
    "input_unit": {
        "type": "input_unit",
        "pooler": {
            "type": "cnn",
            "embedding_dim": 300,
            "num_filters": 300,
            "ngram_filter_sizes": [
                1,
                2,
                3,
                4
            ],
            "output_dim": 300
        },
    },
}