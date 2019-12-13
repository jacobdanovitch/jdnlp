local embeddings = import '../common/embeddings.jsonnet';
local seq2vec = import '../common/seq2vec.jsonnet';

{
    "type": "siamese_triplet_loss",
    "text_field_embedder": embeddings.basic_embedder(25),
    "left_encoder": seq2vec.cnn(25),
}