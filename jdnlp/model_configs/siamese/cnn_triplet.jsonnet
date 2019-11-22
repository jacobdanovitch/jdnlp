{
    "model": {
        "type": "siamese_triplet_loss",
        "text_field_embedder": {
            "token_embedders": {
                "tokens": {
                    "type": "embedding",
                    "embedding_dim": 25,
                    "trainable": true
                }
            }
        },
        "left_encoder": {
            "type": "cnn",
            "embedding_dim": 25,
            "num_filters": 25,
            "ngram_filter_sizes": [
                2,
                3,
                4,
                5
            ]
        }
    },
    "iterator": {
        "type": "basic",
        "batch_size" : 2
    },
    "trainer": {
        "num_epochs": 1,
        "cuda_device": 0,
        "grad_clipping": null,
        "validation_metric": "+loss",
        "optimizer": {
            "type": "adam"
        }
    }
}