local embedding = import '../../common/embeddings.jsonnet';
local seq2vec = import '../../common/seq2vec.jsonnet';

local bert(pretrained_model='bert-base-uncased', trainable=false) = {
    "text_field_embedder": embedding.bert(pretrained_model, trainable),
    "input_unit": {
        "type": "input_unit",
        "pooler": seq2vec.bert_pooler(pretrained_model)
    }
};

local transformer(pretrained_model) = {
    "text_field_embedder": embedding.transformer(pretrained_model),
    "input_unit": {
        "type": "input_unit",
        "pooler": seq2vec.bert_pooler('bert-base-uncased') //pretrained_model
    }
};

{
    bert::bert,
    transformer::transformer
}