local BASE_ITERATOR(batch_size=8) = {
    "type": "basic",
    "batch_size" : batch_size
};

local MULTIPROCESS_ITERATOR = {
    "type": "multiprocess", 
    "base_iterator": BASE_ITERATOR(),
    "num_workers": 16
};

{
    base_iterator(batch_size)::
        BASE_ITERATOR(batch_size)
}