local common = import '../experiments/common/index.jsonnet'; // from run/root directory (jdnlp)
local model = import '../jdnlp/model_configs/siamese/cnn_triplet.jsonnet';

local embeddings = import '../jdnlp/model_configs/common/embeddings.jsonnet';
local seq2vec = import '../jdnlp/model_configs/common/seq2vec.jsonnet';
local seq2seq = import '../jdnlp/model_configs/common/seq2seq.jsonnet';

local embedding_dim = 100;

local base_model(enc) = {
    "type": "siamese_triplet_loss",
    "text_field_embedder": embeddings.basic_embedder(embedding_dim, pretrained='glove'),
    "encoder": {
        "type": "pooling",
        "encoder": enc
    }
};

local cnn = {
    "type": "siamese_triplet_loss",
    "text_field_embedder": embeddings.basic_embedder(embedding_dim, pretrained='glove'),
    "encoder": seq2vec.cnn(embedding_dim)
};

local gru = base_model(seq2seq.rnn(embedding_dim, 'gru'));
local han = base_model(seq2seq.han(embedding_dim));
local sparse_transformer = base_model({
    "type": "sparse_transformer",
    "emb": embedding_dim,
    "heads": 1
});


{
    "dataset_reader": {
        "type": "triplet_reader", //"nnm_reader",
        "comment_index_path": "datasets/nnm/comment_index.json",
        "article_index_path": "datasets/nnm/text_index.json",
        "max_seq_len": 10000,
        "sample": 1000

    },
    "train_data_path": "datasets/nnm/train_triplets.csv",
    // "validation_data_path": "datasets/nnm/test_sample.csv",
    
    "model": gru, // sparse_transformer,
    
    "iterator": common.iterators.base_iterator(batch_size=16),
    "trainer": common.trainer('adam', 0.001, num_epochs=3, validation_metric='-loss')
}
