local seq2vec = import 'seq2vec.jsonnet';

// https://nlp.h-its.org/bpemb/#download
// https://fasttext.cc/docs/en/english-vectors.html
local PRETRAINED = {
    "glove": {
        "100": "https://allennlp.s3.amazonaws.com/datasets/glove/glove.6B.100d.txt.gz",
        "300": "https://allennlp.s3.amazonaws.com/datasets/glove/glove.840B.300d.txt.gz",
    }
};

local BASIC_EMBEDDER(dim, trainable=true, pretrained=null) = {
    "token_embedders": {
        "tokens": {
            "type": "embedding",
            "embedding_dim": dim,
            "trainable": trainable,
            [if pretrained != null then 'pretrained_file']: PRETRAINED[pretrained][''+dim],
        }
    }
};

local CHAR_EMBEDDER(dim, trainable=true, pretrained=null, char_dim=50, dropout=0.4, tokens=true) = {
    "token_embedders": {
        [if tokens then "tokens"]: {
            "type": "embedding",
            "embedding_dim": dim,
            "trainable": trainable,
            [if pretrained != null then 'pretrained_file']: PRETRAINED[pretrained][''+dim],
            "projection_dim": dim/2
        },
        "token_characters": {
            "type": "character_encoding",
            "embedding": {
                "embedding_dim": char_dim
            },
            "encoder": seq2vec.cnn(char_dim, output_dim=if tokens then dim/2 else dim),
            "dropout": dropout
        }
    }
};

// https://allennlp.org/elmo
// https://github.com/allenai/allennlp/blob/master/tutorials/how_to/training_transformer_elmo.md
local ELMO_EMBEDDER(dim=null, dropout=0.5, requires_grad=false) = {
        "elmo": {
            "type": "elmo_token_embedder",
            "options_file": "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x1024_128_2048cnn_1xhighway/elmo_2x1024_128_2048cnn_1xhighway_options.json",
            "weight_file": "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x1024_128_2048cnn_1xhighway/elmo_2x1024_128_2048cnn_1xhighway_weights.hdf5",
            "do_layer_norm": false,
            "requires_grad": requires_grad,
            "dropout": dropout,
            [if dim != null then "projection_dim"]: dim,
        }
};

local BERT_EMBEDDER(pretrained_model="bert-base-uncased", trainable=false) = {
    "tokens": {
        "type": "bert-pretrained",
        "pretrained_model": pretrained_model,
        "requires_grad": trainable,
        "top_layer_only": false
    }, 
    "allow_unmatched_keys": true
};

local TRANSFORMER_EMBEDDER(pretrained_model) = {
    "tokens": {
        "type": "pretrained_transformer",
        "model_name": pretrained_model
    },
    "allow_unmatched_keys": true
};

{
    basic_embedder::BASIC_EMBEDDER,
    char_embedder::CHAR_EMBEDDER,
    elmo_embedder::ELMO_EMBEDDER,
    bert::BERT_EMBEDDER,
    transformer::TRANSFORMER_EMBEDDER
}
