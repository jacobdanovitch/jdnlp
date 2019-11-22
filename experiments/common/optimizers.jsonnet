{
    adam(lr=3e-4)::{
        "type": "adam",
        "lr": lr
    },
    bert(lr=5e-5)::{
        "type": "bert_adam",
        "lr": lr,
        "t_total": -1,
        "max_grad_norm": 1.0,
        "weight_decay": 0.01,
        "parameter_groups": [
            [["bias", "LayerNorm.bias", "LayerNorm.weight", "layer_norm.weight"], {"weight_decay": 0.0}],
        ],
    }
}