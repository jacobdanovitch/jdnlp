local BASE_ITERATOR(batch_size=8) = {
    "type": "basic",
    "batch_size" : batch_size
};

local BUCKET(batch_size, sorting_keys, max_instances_in_memory=null, skip_smaller_batches=false) = {
    "type": "bucket",
    "batch_size": batch_size,
    [if max_instances_in_memory != null then "max_instances_in_memory"]: max_instances_in_memory,

    "skip_smaller_batches": skip_smaller_batches,

    "sorting_keys": sorting_keys,
    "biggest_batch_first": true,
};

local MULTIPROCESS_ITERATOR = {
    "type": "multiprocess", 
    "base_iterator": BASE_ITERATOR(),
    "num_workers": 16
};

{
    base_iterator::BASE_ITERATOR,
    bucket_iterator::BUCKET
}