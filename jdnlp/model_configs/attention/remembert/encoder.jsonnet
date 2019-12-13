local seq2vec = import '../../common/seq2vec.jsonnet';
local seq2seq = import '../../common/seq2seq.jsonnet';

local encoder(pooler, context_encoder=null, knowledge_encoder=null) = {
    "type": "input_unit",
    "pooler": pooler,
    [if context_encoder != null then 'context_encoder']: context_encoder,
    [if knowledge_encoder != null then 'knowledge_encoder']: knowledge_encoder // 
};


local prebuilt(pooler) = function(dim) (encoder(pooler));
// context_encoder=seq2seq.han(dim)
//seq2seq.rnn(300, encoder='gru')
//knowledge_encoder=seq2vec.masked_self_attention(300, time_distributed=true)

{
    cnn(dim):: prebuilt(seq2vec.cnn(dim))(dim),
    lstm(dim):: prebuilt(seq2vec.rnn(dim, encoder='lstm'))(dim),
    bilstm(dim):: prebuilt(seq2vec.bilstm(dim))(dim),
    gru(dim):: prebuilt(seq2vec.rnn(dim, encoder='gru'))(dim),
    boe(dim):: prebuilt(seq2vec.boe(dim))(dim),

    build::encoder
}