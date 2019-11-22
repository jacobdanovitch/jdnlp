local embedding = import '../../common/embeddings.jsonnet';

{
    "text_field_embedder": embedding.basic_embedder(300, "glove", true),
    "input_unit": {
        "type": "input_unit",
        "pooler": {
            "type": "lstm",
            "input_size": 300,
            "hidden_size": 300
        },
    },
}