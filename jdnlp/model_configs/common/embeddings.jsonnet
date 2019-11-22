local PRETRAINED = {
    "glove": {
        "300": "https://allennlp.s3.amazonaws.com/datasets/glove/glove.840B.300d.txt.gz",
    }
};

local BASIC_EMBEDDER(dim, pretrained=null, trainable=true) = {
    "token_embedders": {
        "tokens": {
            "type": "embedding",
            "embedding_dim": dim,
            "trainable": trainable,
            [if pretrained != null then 'pretrained_file']: PRETRAINED[pretrained][''+dim],
        }
    }
};

{
    basic_embedder(dim, pretrained=null, trainable=true)::
        BASIC_EMBEDDER(dim, pretrained, trainable)
}
