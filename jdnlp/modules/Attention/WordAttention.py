import torch
import torch.nn as nn
# import torch.nn.functional as F

from torch.nn.utils.rnn import pack_padded_sequence as pack
from torch.nn.utils.rnn import pad_packed_sequence as unpack
from torch.nn.utils.rnn import PackedSequence

from allennlp.modules.seq2vec_encoders import PytorchSeq2VecWrapper
from allennlp.modules.attention import LinearAttention

from jdnlp.modules.Attention.Attention import Attention


class WordAttention(nn.Module):
    def __init__(self, input_size, hidden_size, attention_size, n_layers=1, dropout_p=0, device="cpu"):
        super().__init__()

        self.device=device
        self.input_size = input_size
        self.rnn = nn.GRU(input_size=input_size,
                          hidden_size=hidden_size, #int(hidden_size / 2),
                          num_layers=n_layers,
                          dropout=dropout_p,
                          bidirectional=False,#True,
                          batch_first=True
                          )
        self.rnn = PytorchSeq2VecWrapper(self.rnn)
        self.senlen = 10
        self.attn = LinearAttention(hidden_size, attention_size)


    def forward(self, document, word_per_sentence):
        # print("document:", document.size())
        sentences = document.split(self.senlen, 1)
        # print("sentences:", len(sentences))
        
        outputs, weights = [], []
        for sent in sentences:
            #print("sent:", sent.size())
            
            mask = torch.tensor(sent.size(0) * [sent.size(1)]).unsqueeze(1) # sentence lengths

            hidden = self.rnn(sent, mask)
            hidden = hidden.contiguous()
            #print("hidden:", hidden.size())

            w = self.attn(hidden, hidden)
            #print("weights:", w.size())

            ctx = torch.matmul(w,hidden)
            #print("ctx:", ctx.size())
            outputs.append(ctx)
            weights.append(w)
        
        outputs = torch.stack(outputs, dim=1)
        weights = torch.stack(weights, dim=1)

        return outputs, weights


    

class WordAttention2(nn.Module):
    def __init__(self, input_size, embedding_size, hidden_size, attention_size, n_layers=1, dropout_p=0, device="cpu"):
        super().__init__()

        self.device=device
        self.input_size = input_size
        # self.emb = nn.Embedding(input_size, embedding_size).to(device)
        self.rnn = PytorchSeq2VecWrapper(nn.GRU(input_size=embedding_size,
                          hidden_size=hidden_size, #int(hidden_size / 2),
                          num_layers=n_layers,
                          dropout=dropout_p,
                          bidirectional=False,#True,
                          batch_first=True
                          ))

        self.attn = Attention(hidden_size=hidden_size, attention_size=attention_size)
        #self.attn = LinearAttention(hidden_size, attention_size)


    def forward(self, sentence, word_per_sentence):
        # |sentence| = (sentence_length, max_word_length, embedding_size)

        #print(type(sentence))
        #sentence, _ = unpack(sentence, batch_first=False)
        sentence = pack(sentence, lengths=word_per_sentence, batch_first=True, enforce_sorted=False)
        print("sentence:", sentence.data.size())
        
        mask = Attention.generate_mask(word_per_sentence.data.tolist())
        
        # |last_hiddens| = (sentence_length, max(word_per_sentence), hidden_size)
        last_hiddens = self.rnn(sentence, mask)
        last_hiddens, _ = unpack(last_hiddens, batch_first=True)
        print("hiddens:", last_hiddens.size())
        

        #print("Hidden size:", last_hiddens.size())
        # context_weights = self.attn(last_hiddens, last_hiddens)
        # context_vectors = torch.mm(context_weights, last_hiddens)

        context_vectors, context_weights = self.attn(last_hiddens, mask)
        # |context_vectors| = (sentence_length, hidden_size)
        # |context_weights| = (sentence_length, max(word_per_sentence))

        return context_vectors, context_weights


    def _forward(self, sentence, word_per_sentence):
        # |sentence| = (sentence_length, max_word_length)
        # |word_per_sentence| = (sentence_length)

        # sentence = self.emb(sentence)
        # |sentence| = (sentence_length, max_word_length, embedding_size)
        
        # print(sentence.shape)
        # print(sentence)

        # Pack sentence before insert rnn model.
        
        packed_sentences = pack(sentence,
                                lengths=word_per_sentence.data.tolist(),
                                batch_first=True,
                                enforce_sorted=False)

        # Apply RNN and get hiddens layers of each words
        # print(word_per_sentence.data.size(), packed_sentences.data.size())
        last_hiddens, _ = self.rnn(packed_sentences)

        # Unpack ouput of rnn model
        # last_hiddens, _ = unpack(last_hiddens, batch_first=True)
        # |last_hiddens| = (sentence_length, max(word_per_sentence), hidden_size)

        # Based on the length information, gererate mask to prevent that shorter sample has wasted attention.
        mask = Attention.generate_mask(word_per_sentence)
        # |mask| = (sentence_length, max(word_per_sentence))

        context_vectors, context_weights = self.attn(last_hiddens, mask)
        # |context_vectors| = (sentence_length, hidden_size)
        # |context_weights| = (sentence_length, max(word_per_sentence))

        return context_vectors, context_weights