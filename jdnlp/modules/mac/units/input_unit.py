from typing import Dict, Optional
from overrides import overrides

import torch
import torch.nn as nn

from allennlp.modules import TimeDistributed, Seq2VecEncoder, Seq2SeqEncoder
from allennlp.modules.seq2seq_encoders import PassThroughEncoder
from allennlp.nn import util



import logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.CRITICAL)

@Seq2VecEncoder.register("input_unit")
class BertInputUnit(Seq2VecEncoder):
    def __init__(self, 
    pooler: Seq2VecEncoder,
    encoder: Seq2SeqEncoder = None
    ):
        super().__init__()

        self.encoder = encoder or PassThroughEncoder(pooler.get_input_dim()) 
        # self.encoder = TimeDistributed(self.encoder)
        self.pooler = pooler
        self.output_dim = pooler.get_output_dim()
        
    @overrides
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
        """

        B, T, W, D = conversation.size()
        
        # conversation = self.encoder(conversation, mask=None)#word_mask)
        knowledge, context = conversation.split_with_sizes([T-1, 1], dim=1)
        k_mask, c_mask = word_mask.split_with_sizes([T-1, 1], dim=1)
        
        c_mask = c_mask.squeeze()
        context = self.encoder(context.squeeze(), mask=c_mask)
        
        #logger.warn(c_mask.size())
        #logger.warn(context.size())
        context = context.view(B, W, D) # implicit squeeze here
        question = self.pooler(context, mask=None)#c_mask)
        knowledge = knowledge.view(B, D, (T-1)*W).contiguous()

        # logger.warn(f'Context: {context.size()}')
        # logger.warn(f'Question: {question.size()}')
        # logger.warn(f'Knowledge: {knowledge.size()}')

        return context, question, knowledge


    @overrides
    def get_input_dim(self) -> int:
        return self.input_dim

    @overrides
    def get_output_dim(self) -> int:
        return self.output_dim



