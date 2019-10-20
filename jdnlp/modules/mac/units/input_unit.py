from typing import Dict, Optional
from overrides import overrides

import torch
import torch.nn as nn

from allennlp.modules import TimeDistributed, Seq2VecEncoder, Seq2SeqEncoder
from allennlp.nn import util

from allennlp.modules.token_embedders.bert_token_embedder import PretrainedBertEmbedder
from allennlp.modules.text_field_embedders import BasicTextFieldEmbedder

import logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.CRITICAL)

@Seq2VecEncoder.register("bert_input_unit")
class BertInputUnit(Seq2VecEncoder):
    def __init__(self, 
    pretrained_model: str = 'bert-base-uncased',
    pooler: Seq2VecEncoder = None,
    ):
        super().__init__()

        self.text_field_embedder = BasicTextFieldEmbedder(
            {'tokens': PretrainedBertEmbedder(pretrained_model, top_layer_only=True)},
            allow_unmatched_keys=True
        )

        self.pooler = pooler # TimeDistributed(pooler)

        self.output_dim = pooler.get_output_dim()
        
    @overrides
    def forward(self, conversation: torch.Tensor, word_mask: torch.Tensor):
        """
        Parameters
        ----------
        conversation : Dict[str, Variable], required
            ListField of shape [B, T, W].
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
        e_conv = self.text_field_embedder(conversation)
        # logger.warn(f'Conv: {e_conv.size()}')

        B, T, W, D = e_conv.size()
        
        knowledge, context = e_conv.split_with_sizes([T-1, 1], dim=1)
        
        context = context.view(B, W, D) # implicit squeeze here
        question = self.pooler(context.squeeze(), word_mask)
        knowledge = knowledge.view(B, D, (T-1)*W)

        # logger.warn(f'Ctx: {context.size()}')
        # logger.warn(f'Question: {question.size()}')
        # logger.warn(f'Knowledge: {knowledge.size()}')

        return context, question, knowledge


    @overrides
    def get_input_dim(self) -> int:
        return self.input_dim

    @overrides
    def get_output_dim(self) -> int:
        return self.output_dim



