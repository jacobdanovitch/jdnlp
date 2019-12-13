local common = import '../experiments/common/index.jsonnet'; // from run/root directory (jdnlp)
local model = import '../jdnlp/model_configs/siamese/cnn_triplet.jsonnet';

local embeddings = import '../jdnlp/model_configs/common/embeddings.jsonnet';
local seq2vec = import '../jdnlp/model_configs/common/seq2vec.jsonnet';
local seq2seq = import '../jdnlp/model_configs/common/seq2seq.jsonnet';

local embedding_dim = 300; // 100

local base_model(enc) = {
    "type": "siamese_triplet_loss",
    // "text_field_embedder": embeddings.char_embedder(embedding_dim, tokens=false),
    "text_field_embedder": embeddings.basic_embedder(embedding_dim, pretrained='glove'),
    //"text_field_embedder": embeddings.elmo_embedder(dim=embedding_dim, requires_grad=true),
    "encoder": enc,
    "loss_margin": 0.15
};

local pooling_model(enc) =
    base_model({
        "type": "pooling",
        "encoder": enc,
    });

local han = {
    "type": "siamese_triplet_loss",
    "text_field_embedder": embeddings.basic_embedder(embedding_dim, pretrained=null),#'glove'),
    "encoder": {
        "type": "han_encoder",
        "sentence_encoder": seq2vec.gru(embedding_dim),
        "document_encoder":seq2vec.gru(embedding_dim),
        "word_attention": seq2seq.han(embedding_dim),
        "sentence_attention": seq2seq.han(embedding_dim)
    },
    "loss_margin": 5
};

local sparse_transformer = pooling_model({
    "type": "sparse_transformer",
    "emb": embedding_dim,
    "heads": 1
});


local adaptive_transformer = pooling_model({
    "type": "adaptive_transformer",
    'nb_layers': 1,
    
    // seqlayer
    "hidden_size": embedding_dim, 
    'nb_heads': 4,  
    'attn_span': 150, // up this
    'inner_hidden_size': embedding_dim*4,

    'dropout': 0.01,
    'adapt_span_params': {
        'adapt_span_enabled': true,
        'adapt_span_loss': 0.0000005, 
        'adapt_span_ramp': 0.2, 
        'adapt_span_init': 0.1,
        'adapt_span_cache': true, 
    }
});

local pretrained_adaptive_transformer = {
    "type": "siamese_triplet_loss",
    "text_field_embedder": embeddings.char_embedder(512, tokens=false),
    "encoder": {
        "type": "pooling",
        "encoder": {
            "type": "adaptive_transformer",
            'nb_layers': 12,
            
            // seqlayer
            "hidden_size": 512, 
            'nb_heads': 8,  
            'attn_span': 8192, // up this
            'inner_hidden_size': 2048,

            'dropout': 0.01,
            'adapt_span_params': {
                'adapt_span_enabled': true,
                'adapt_span_loss': 0.0000005, 
                'adapt_span_ramp': 0.2, 
                'adapt_span_init': 0.1,
                'adapt_span_cache': true, 
            },

            'pretrained_fp': '/home/jacobgdt/.pretrained/enwik8.pt'
        },
    },
    "loss_margin": 0.15
};

local biblosa = base_model({
    "type": "block-self-attention",
    "input_dim": embedding_dim,
    "expected_max_length": 100
});

local boe = {
    "type": "siamese_triplet_loss",
    "text_field_embedder": embeddings.basic_embedder(100, false, pretrained='glove'),
    "encoder": seq2vec.boe(100)
};

{
    "dataset_reader": {
        "type": "tweetlink_reader", //"nnm_reader",
        // /*
        "token_indexers": {
            // /* 
            "tokens": {
                "type": "single_id",
                "lowercase_tokens": true
                },
            // */
            /*
            "token_characters": {
                "type": "characters"
            }
            */
            /*
            "elmo": {
                "type": "elmo_characters"
            }
            */
        },
        // */
        "sample": 5000,
    },
    "train_data_path": "datasets/tweet-linking/tweetlinking.jsonl",
    //"train_data_path": "datasets/reddit-linking/titleonly_100k.jsonl",
    
    // "model": sparse_transformer,
    "model": base_model(seq2vec.bigru(embedding_dim)), //
    // "model": base_model(seq2vec.cnn(embedding_dim)),
    // "model": adaptive_transformer,
    // "model": pretrained_adaptive_transformer,
    
    // "iterator": common.iterators.bucket_iterator(batch_size=8, sorting_keys=[['anchor', 'num_token_characters']], skip_smaller_batches=true),
    "iterator": common.iterators.base_iterator(batch_size=32),
    "trainer": common.trainer('adam', lr=0.001, num_epochs=3) //5)
}