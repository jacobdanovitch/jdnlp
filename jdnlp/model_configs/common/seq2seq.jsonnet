local rnn(dim, encoder, bidirectional=false) = {
    "type": encoder,
    "input_size": dim,
    "hidden_size": if bidirectional then dim / 2 else dim,
    "bidirectional": bidirectional
};

local han(dim, context_dim=null) = {
    "type": "simple_han_attention",
    "input_dim": dim,
    "context_vector_dim": if context_dim != null then context_dim else dim,
};




{
    rnn::rnn,
    han::han
}