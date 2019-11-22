local EMBEDDING_DIM = 300;

{
    "model": {
        "type": "siamese_triplet_loss",
        "text_field_embedder": {
            "token_embedders": {
                "tokens": {
                    "type": "embedding",
                    "embedding_dim": EMBEDDING_DIM,
                    "trainable": true,
                    "pretrained_file": "https://s3-us-west-2.amazonaws.com/allennlp/datasets/glove/glove.840B.300d.txt.gz",
                }
            }
        },
        // "prediction_threshold": 0.5,
        "left_encoder": {
            "type": "lstm",
            "input_size": EMBEDDING_DIM,
            "hidden_size": EMBEDDING_DIM,
            "num_layers": 1
        },
        "right_encoder": {
            "type": "lstm",
            "input_size": EMBEDDING_DIM,
            "hidden_size": EMBEDDING_DIM,
            "num_layers": 1
        }
    },
    "iterator": {
        "type": "basic",
        "batch_size" : 16
    },
    "trainer": {
        "num_epochs": 3,
        "cuda_device": 0,
        "grad_clipping": null,
        "validation_metric": "+accuracy",
        "optimizer": {
            "type": "adam"
        }
    }
}