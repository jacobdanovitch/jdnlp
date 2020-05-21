local pooling(encoder) = {
    "type": "pooling",
    "encoder": encoder,
};

local cnn(dim, filters=[1,2,3,4], output_dim=null) = {
    "type": "cnn",
    "embedding_dim": dim,
    "num_filters": dim,
    "ngram_filter_sizes": filters,
    "output_dim": if output_dim != null then output_dim else dim
};

local rnn(dim, encoder, num_layers=1) = {
    "type": encoder,
    "input_size": dim,
    "hidden_size": dim,
    "num_layers": num_layers
};

local birnn(dim, encoder, num_layers=1) = {
    "type": encoder,
    "input_size": dim,
    "hidden_size": dim/2,
    "num_layers": num_layers,
    "bidirectional": true
};

local boe(dim) = {
    "type": "boe",
    "embedding_dim": dim
};

local masked_self_attn(dim, encoder='gru', time_distributed=false) = {
    "type": "masked_self_attention",
    "encoder": rnn(dim, encoder),
    "time_distributed": time_distributed
};

local multi_head_self_attention(dim, num_heads=1) = {
    "type": "multi_head_self_attention",
    "num_heads": num_heads,
    "input_dim": dim,
    "attention_dim": dim,
    "values_dim": dim,
    "output_projection_dim": dim
};

local bert_pooler(pretrained_model="bert-base-uncased", requires_grad=true) = {
    "type": "bert_pooler",
    "pretrained_model": pretrained_model,
    "requires_grad": requires_grad
};

local star_transformer(dim, num_layers=1, num_head=3, unfold_size=3, dropout=0.1) = {
    "type": "star_transformer",
    "hidden_size": dim, 
    "num_layers": num_layers, 
    "num_head": num_head, 
    "head_dim": dim/num_head,
    "unfold_size": unfold_size,
    "dropout": dropout
};

{
    pooling::pooling,
    boe::boe,
    cnn:: cnn,
    
    rnn:: rnn,
    gru:: (function(dim) rnn(dim, encoder='gru')),
    bigru:: (function(dim) birnn(dim, encoder='gru')),
    lstm:: (function(dim) rnn(dim, encoder='lstm')),
    bilstm:: (function(dim) birnn(dim, encoder='lstm')),

    masked_self_attention:: masked_self_attn,
    multi_head_self_attention:: multi_head_self_attention,
    bert_pooler::bert_pooler,
    
    star_transformer::star_transformer
}