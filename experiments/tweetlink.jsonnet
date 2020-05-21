local common = import '../experiments/common/index.jsonnet'; // from run/root directory (jdnlp)

local embeddings = import '../jdnlp/model_configs/common/embeddings.jsonnet';
local seq2vec = import '../jdnlp/model_configs/common/seq2vec.jsonnet';
local seq2seq = import '../jdnlp/model_configs/common/seq2seq.jsonnet';

local embedding_dim = 300; // 100
local base_model(enc) = {
    "type": "siamese_triplet_loss",
    "text_field_embedder": embeddings.basic_embedder(embedding_dim, pretrained='glove'),
    //"text_field_embedder": embeddings.basic_embedder(embedding_dim),
    "encoder": enc,
    "loss_margin": 0.075
};
local pooling_model(enc) = base_model(seq2vec.pooling(enc));

local datasets = {
    "reddit_titles": "datasets/reddit-linking/titleonly_100k.jsonl",
    "reddit": "datasets/reddit-linking/train.jsonl",
    "twitter": "datasets/tweet-linking/train.jsonl",
};

local models = {
    biblosa: pooling_model(seq2seq.biblosa(embedding_dim)),
    sparse_transformer: pooling_model(seq2seq.sparse_transformer(embedding_dim)),
    adaptive_transformer: pooling_model(seq2seq.adaptive_transformer(embedding_dim)),
    pretrained_adaptive_transformer: {
        "type": "siamese_triplet_loss",
        "text_field_embedder": embeddings.basic_embedder(512),
        "encoder": {
            "type": "pooling",
            "encoder": seq2seq.pretrained_adaptive_transformer(),
        },
        "loss_margin": 0.15
    },

    attn: pooling_model(seq2seq.learned_attn(embedding_dim)),
    star_transformer: base_model(seq2vec.star_transformer(embedding_dim, dropout=0.0, num_layers=1, num_head=10, unfold_size=5)), // 1L, 1H, DO=0.1 worked ok
    
    gru: base_model(seq2vec.gru(embedding_dim)),
    bigru: base_model(seq2vec.bigru(embedding_dim)),
    cnn: base_model(seq2vec.cnn(embedding_dim))
};

local dataset = std.extVar('data');
local model = std.extVar('model');

{
    "dataset_reader": {
        "type": "tweetlink_reader",
        "token_indexers": {
            "tokens": {
                "type": "single_id",
                "lowercase_tokens": true
                }
        },
        //"max_sequence_length": 1000
        // "sample": 5000, //20000,
    },

    "train_data_path": datasets[dataset],
    "model": models[model],
    
    // "iterator": common.iterators.bucket_iterator(batch_size=8, sorting_keys=[['anchor', 'num_token_characters']], skip_smaller_batches=true),
    "iterator": common.iterators.base_iterator(batch_size=128),
    "trainer": common.trainer('adam', lr=0.001, num_epochs=10, cuda_device=[0, 1, 2, 3], patience=5)
}