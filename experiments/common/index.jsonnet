local iterators = import 'iterators.jsonnet';
local optimizers = import 'optimizers.jsonnet';
local utils = import 'utils.jsonnet';

{
    "iterators": iterators,
    "optimizers": optimizers,
    "utils": utils,

    trainer(optimizer, lr=1e-4, lr_schedule=null)::{
        "num_epochs": 10,
        "cuda_device": 0,
        "optimizer": optimizers[optimizer](lr),
        [if lr_schedule != null then 'learning_rate_scheduler']: lr_schedule
    }
}