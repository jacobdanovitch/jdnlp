local input_units = import 'remembert/index.jsonnet';

local embedding = import '../common/embeddings.jsonnet';
local seq2vec = import '../common/seq2vec.jsonnet';
local seq2seq = import '../common/seq2seq.jsonnet';

local BASE_DIM = 300;

local knowledge_encoder(dim) = {"knowledge_encoder": seq2seq.rnn(dim, 'gru') + {"bidirectional": true, "hidden_size": dim/2}};

local Config = {
    encoder(model, dim, trainable_embeddings=true, pretrained=null):: {
        //"text_field_embedder": embedding.char_embedder(dim, trainable_embeddings, pretrained),
        "text_field_embedder": embedding.basic_embedder(dim, trainable_embeddings, pretrained),
        "input_unit": input_units.encoder[model](dim),
    },
    bert: input_units.bert.bert(),
    distilbert: input_units.bert.transformer('distilbert-base-uncased'),
    // kb(),
};


// used do=0.7, max_step=4, n_mem=2, selfattn=false for emocontext exps
//local MACNet(input_unit, max_step=4, n_memories=2, self_attention=false, memory_gate=true, dropout=0.7) = {
local MACNet(input_unit, max_step=2, n_memories=1, self_attention=false, memory_gate=true, dropout=0.8) = {
    "model": {
        "type": "mac_network",

        "max_step": max_step,
        "n_memories": n_memories,
        
        "self_attention": self_attention,
        "memory_gate": memory_gate,
        "dropout": dropout
    } + input_unit
};

{
    model(enc, dim, pretrained='glove')::MACNet(Config.encoder(enc, dim, pretrained)),

    dev: MACNet(Config.encoder('boe', 100, pretrained=null)), 
    cnn: MACNet(Config.encoder('cnn', BASE_DIM, pretrained='glove')),
    gru: MACNet(Config.encoder('gru', BASE_DIM, pretrained='glove')),
    lstm: MACNet(Config.encoder('lstm', BASE_DIM, pretrained='glove')),
    bilstm: MACNet(Config.encoder('bilstm', BASE_DIM, pretrained='glove')),
    boe: MACNet(Config.encoder('boe', BASE_DIM, pretrained='glove')),
    bert: MACNet(Config.bert)
}