local rnn(dim, encoder, bidirectional=false) = {
    "type": encoder,
    "input_size": dim,
    "hidden_size": if bidirectional then dim / 2 else dim,
    "bidirectional": bidirectional
};

local learned_attn(dim, context_dim=null) = {
    "type": "simple_han_attention",
    "input_dim": dim,
    "context_vector_dim": if context_dim != null then context_dim else dim,
};

local dcn(dim, k_max_number=100) = {
    "type": "dynamic_conv",
    "input_dim": dim,
    "output_dim": dim,
    "cells": [
        {'cell_number': 1,
            'sent_length': 7,
            'conv_kernel_size': [3, 1],
            'conv_input_channels': 1,
            'conv_output_channels': 1,
            'conv_stride': [1, 1],
            'k_max_number': k_max_number,
            'folding_kernel_size': [1, 2],
            'folding_stride': [1, 1]
        },
        {'cell_number': -1,
            'sent_length': 7,
            'conv_kernel_size': [3, 1],
            'conv_input_channels': 1,
            'conv_output_channels': 2,
            'conv_stride': [1, 1],
            'k_max_number': k_max_number,
            'folding_kernel_size': [1, 2],
            'folding_stride': [1, 1]
        }
    ]
};

local biblosa(dim, expected_max_length=100) = {
    "type": "block-self-attention",
    "input_dim": dim,
    "expected_max_length": expected_max_length
};

local adaptive_transformer(dim, attn_span=1000) = {
    "type": "adaptive_transformer",
    'nb_layers': 1,
    
    // seqlayer
    "hidden_size": dim, 
    'nb_heads': 4,  
    'attn_span': attn_span, // 8192 in base
    'inner_hidden_size': dim*4,

    'dropout': 0.01,
    'adapt_span_params': {
        'adapt_span_enabled': true,
        'adapt_span_loss': 0.0000005, 
        'adapt_span_ramp': 0.2, 
        'adapt_span_init': 0.1,
        'adapt_span_cache': true, 
    }
};

local pretrained_adaptive_transformer() = {
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
};

local sparse_transformer(dim, heads=1) = {
    "type": "sparse_transformer",
    "emb": dim,
    "heads": heads
};

local star_transformer(dim, num_layers=1, num_head=1, dropout=0.1) = {
    "type": "star_transformer",
    "hidden_size": dim, 
    "num_layers": num_layers, 
    "num_head": num_head, 
    "head_dim": dim,
    "dropout": dropout
};


{
    rnn::rnn,
    learned_attn::learned_attn,
    dcn::dcn,
    biblosa::biblosa,
    adaptive_transformer::adaptive_transformer,
    pretrained_adaptive_transformer::pretrained_adaptive_transformer,
    sparse_transformer::sparse_transformer,
    star_transformer::star_transformer
}