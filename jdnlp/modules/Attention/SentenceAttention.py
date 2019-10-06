import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn.utils.rnn import pack_padded_sequence as pack
from torch.nn.utils.rnn import pad_packed_sequence as unpack
from torch.nn.utils.rnn import PackedSequence

from allennlp.modules.similarity_functions import LinearSimilarity
from allennlp.modules.seq2vec_encoders import PytorchSeq2VecWrapper
from allennlp.modules.attention import LinearAttention, BilinearAttention, AdditiveAttention
from allennlp.modules import Seq2VecEncoder

from jdnlp.modules.Attention.Attention import Attention

# Seq2VecEncoder
class SentenceAttention(nn.Module):
    def __init__(self, input_size, hidden_size, attention_size, n_layers=1, dropout_p=0, device="cpu"):
        super().__init__()

        self.device=device
        self.hidden_size = hidden_size
        self.rnn = nn.GRU(input_size=input_size,
                          hidden_size=hidden_size, #int(hidden_size / 2),
                          num_layers=n_layers,
                          dropout=dropout_p,
                          bidirectional=False,#True,
                          batch_first=True
                          )
        # self.rnn = PytorchSeq2VecWrapper(self.rnn)
        # self.attn = Attention(hidden_size=hidden_size, attention_size=attention_size)
        # self.attn = LinearAttention(hidden_size, 0)
        # self.sim = LinearSimilarity(hidden_size, attention_size)
        self.proj = nn.Linear(hidden_size, hidden_size)
        self.context_vector = nn.Parameter(torch.randn(attention_size).float())
        self.softmax = nn.Softmax(dim=0)


    
    def forward(self, sentences, sentence_per_document):
        # sentences = [batch_size, sen_len, hidden_size]

        mask = torch.tensor(sentences.size(0) * [sentences.size(1)]).unsqueeze(1) # sentence lengths
        
        hidden, _ = self.rnn(sentences)
        #print("hidden:", hidden.size())

        hidden_projection = torch.tanh(self.proj(hidden))
        #print("hidden proj:", hidden_projection.size(), self.context_vector.size())
        score = hidden_projection.matmul(self.context_vector)
        attn = self.softmax(score)
        #print("attn:", attn.size())

        output = attn.matmul(hidden).sum(dim=1)

        #grad = self.context_vector.grad
        #print(grad is not None and grad.size())
        return output, attn


        #ctx = hidden
        #w = None

        # w = self.attn(hidden, hidden)
        # w = torch.matmul(self.sim(hidden, hidden), hidden)
        #print("weights:", w.size())

        # ctx = torch.matmul(w,hidden)
        # ctx = hidden * w
        #print("ctx:", ctx.size())

        # return ctx, w
    