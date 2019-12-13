from typing import Dict, Optional, Tuple
from overrides import overrides

import torch
import torch.nn as nn

from allennlp.modules import TimeDistributed, Seq2VecEncoder, Seq2SeqEncoder
from allennlp.modules.seq2seq_encoders import PassThroughEncoder
from allennlp.nn import util
from allennlp.modules.matrix_attention import CosineMatrixAttention, DotProductMatrixAttention

import torchsnooper

import logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.CRITICAL)


@Seq2VecEncoder.register("input_unit")
class InputUnit(Seq2VecEncoder):
    def __init__(self, 
    pooler: Seq2VecEncoder,
    knowledge_encoder: Seq2SeqEncoder = None
    ):
        super().__init__()
        self.pooler = pooler
        pass_thru = PassThroughEncoder(pooler.get_input_dim())
        
        self.knowledge_encoder = TimeDistributed(knowledge_encoder or pass_thru) # TimeDistributed(context_encoder)
        
        self.knowledge_attn = DotProductMatrixAttention() # CosineMatrixAttention()
        # self.attn = DotProductMatrixAttention()
        
        self.input_dim = pooler.get_input_dim()
        self.output_dim = pooler.get_output_dim()
    
    # @torchsnooper.snoop()
    def knowledge_self_attention(self, k, k_mask, do_sum=False):
        if k.dim() > 3:
            B, T, W, D = k.size()
            k = k.view(B, T*W, D)
            k_mask = k_mask.view(B, T*W)
        attn = self.knowledge_attn(k, k)
        attn = util.masked_softmax(attn, k_mask, memory_efficient=True)
        
        if k.dim() >= 3:
            k = k.contiguous().view(B*T, W, D)
            attn = attn.view(B*T, W, W*2)
        
        logger.warn(k.shape)
        k = torch.bmm(attn, k)
        
        if do_sum:
            k = k.sum(dim=-2)
        return k.squeeze()

    # @torchsnooper.snoop()#watch=('knowledge.shape', 'context.shape', 'question.shape')) #@overrides
    def forward(self, conversation: torch.Tensor, word_mask: torch.Tensor, turn_mask: torch.Tensor):
        """
        Parameters
        ----------
        conversation : Dict[str, Variable], required
            Tensor of shape [B, T, W, D].
        label : Variable, optional (default = None)
            A variable representing the label for each instance in the batch.

        Returns
        -------
        context:    
            all words of the final turn     [BS, W, D]
        question:   
            [CLS] of final turn             [BS, D]
        knowledge:  
            previous 2 turns                [BS, D, (T-1)*n_words]
            if condensed repr:              [BS, D, T-1]
        """
        
        """
        Split conversation to take last turn as context.
        Knowledge:  [B, T-1, W, D]
        Context:    [B, 1, W, D]
        """
        B, T, W, D = conversation.size()
        
        conversation = self.knowledge_encoder(conversation, word_mask)
        knowledge, context = conversation.split_with_sizes([T-1, 1], dim=1)
        k_mask, c_mask = word_mask.split_with_sizes([T-1, 1], dim=1)
        
        c_mask = c_mask.squeeze()
        context = context.squeeze().contiguous()
        knowledge = knowledge.view(B, D, -1).contiguous() # .sum(dim=-2)
        

        question = self.pooler(context, mask=c_mask)

        return context, question, knowledge


    @overrides
    def get_input_dim(self) -> int:
        return self.input_dim

    @overrides
    def get_output_dim(self) -> int:
        return self.output_dim



@Seq2VecEncoder.register("kb_input_unit")
class KBInputUnit(Seq2VecEncoder):
    def __init__(self, 
    pooler: Seq2VecEncoder,
    context_encoder: Seq2SeqEncoder = None,
    kb_path: str = None,
    kb_shape: Tuple[int, int] = None,
    trainable_kb: bool = False,
    projection_dim: int = None
    ):
        super().__init__()

        kb = (torch.load(kb_path) if kb_path else torch.ones(kb_shape)).float()
        self.knowledge = nn.Parameter(kb, requires_grad=trainable_kb).float()
        self.projection_dim = projection_dim
        if projection_dim:
            self.kb_proj = nn.Linear(self.knowledge.size(0), self.projection_dim)
        
        
        self.context_encoder = context_encoder or PassThroughEncoder(pooler.get_input_dim()) 
        self.pooler = pooler
        self.output_dim = pooler.get_output_dim()
        
    def load_knowledge(self, batch_size):
        kb = self.knowledge
        if self.projection_dim:
            kb = self.kb_proj(kb.t()).t()
        kb = kb.repeat(batch_size, 1, 1)
        return kb
    
    @overrides
    def forward(self, text: torch.Tensor, mask: torch.Tensor):
        """
        Parameters
        ----------
        text : Dict[str, Variable], required
            Tensor of shape [B, W, D].
        label : Variable, optional (default = None)
            A variable representing the label for each instance in the batch.

        Returns
        -------
        context:    
            all words of the text       [BS, W, D]
        question:   
            [CLS] of text               [BS, D]
        knowledge:  
            static KB                   [BS, D, L]
        """

        
        # context = text
        context =  self.context_encoder(text, mask=mask)
        question = self.pooler(context, mask=mask)
        # question = util.get_final_encoder_states(context, mask)
        knowledge = self.load_knowledge(context.size(0))

        return context, question, knowledge


    @overrides
    def get_input_dim(self) -> int:
        return self.input_dim

    @overrides
    def get_output_dim(self) -> int:
        return self.output_dim
    