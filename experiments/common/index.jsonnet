local iterators = import 'iterators.jsonnet';
local optimizers = import 'optimizers.jsonnet';
local utils = import 'utils.jsonnet';

{
    "iterators": iterators,
    "optimizers": optimizers,
    "utils": utils,

    trainer(optimizer, lr=1e-4, num_epochs=10, lr_schedule=null, validation_metric="+accuracy", patience=3, cuda_device=0)::{
        "num_epochs": num_epochs,
        "cuda_device": cuda_device,
        "optimizer": optimizers[optimizer](lr),
        // "num_serialized_models_to_keep": 1,
        [if lr_schedule != null then 'learning_rate_scheduler']: lr_schedule,
        [if validation_metric != null then 'validation_metric']: validation_metric,
        [if patience != null then "patience"]: patience
    }
}