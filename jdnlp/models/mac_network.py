# https://github.com/ivegner/Multi-Memory-MAC-Network
# https://github.com/tohinz/pytorch-mac-network/blob/master/code/mac.py
# https://github.com/rosinality/mac-network-pytorch/blob/master/model.py

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy

from typing import Dict, Optional
from overrides import overrides

from allennlp.common.checks import ConfigurationError
from allennlp.data import Vocabulary

from allennlp.models.model import Model
from allennlp.modules import FeedForward, Seq2VecEncoder, Seq2SeqEncoder, TextFieldEmbedder, TimeDistributed

from allennlp.modules.text_field_embedders import BasicTextFieldEmbedder

from allennlp.nn import InitializerApplicator, RegularizerApplicator
from allennlp.nn import util, Activation
from allennlp.training.metrics import CategoricalAccuracy, FBetaMeasure, F1Measure, Metric

from jdnlp.modules.mac.mac_cell import MACCell
from jdnlp.metrics import WeightedF1Measure

import torchsnooper
import logging
logger = logging.getLogger(__name__)

@Model.register("mac_network")
class MACNetwork(Model):
    """
    Parameters
    ----------
    vocab : ``Vocabulary``, required
        A Vocabulary, required in order to compute sizes for input/output projections.
    text_field_embedder : ``TextFieldEmbedder``, required
        Used to embed the ``tokens`` ``TextField`` we get as input to the model.
    encoder : ``Seq2VecEncoder``
        The encoder that we will use to convert the title to a vector.
    classifier_feedforward : ``FeedForward``
        The fully connected output layer.
    loss_weights: ``torch.Tensor``
        A tensor that determines the weight of each class within the loss function.
    initializer : ``InitializerApplicator``, optional (default=``InitializerApplicator()``)
        Used to initialize the model parameters.
    regularizer : ``RegularizerApplicator``, optional (default=``None``)
        If provided, will be used to calculate the regularization penalty during training.
    """
    def __init__(self, 
                 vocab: Vocabulary,
                 input_unit: Seq2VecEncoder,
                 
                 text_field_embedder: TextFieldEmbedder,
                 # embedding_projection_dim: int = None,
                 
                 classifier_feedforward: FeedForward = None,
                 max_step: int = 12,
                 n_memories: int = 3,
                 self_attention: bool = False,
                 memory_gate: bool = False,
                 dropout: int = 0.15,
                 
                 loss_weights=None,
                 
                 initializer: InitializerApplicator = InitializerApplicator(),
                 regularizer: Optional[RegularizerApplicator] = None
                 ) -> None:
        super().__init__(vocab, regularizer)

        self.num_classes = max(self.vocab.get_vocab_size("labels"), 2)
        
        self.text_field_embedder = text_field_embedder
        
        self.proj = nn.Linear(text_field_embedder.get_output_dim(), input_unit.get_input_dim())
        self.input_unit = input_unit
        self.mac = MACCell(
            text_field_embedder.get_output_dim(), # input_unit.get_output_dim(),
            max_step=max_step,
            n_memories=n_memories,
            self_attention=self_attention,
            memory_gate=memory_gate,
            dropout=dropout,
            save_attns=False,
        )

        hidden_size = 2 * input_unit.get_output_dim()
        n_layers = 3
        self.classifier = classifier_feedforward or FeedForward(
            input_dim = hidden_size,
            num_layers = n_layers,
            hidden_dims = (n_layers-1) * [hidden_size] + [self.num_classes],
            activations = [
                Activation.by_name("relu")(),
                Activation.by_name("relu")(),
                Activation.by_name("linear")()
            ],
            dropout = [
                dropout,
                dropout,
                0.0
            ]
        )

        self.metrics = {
                "accuracy": CategoricalAccuracy(),
                "f1": F1Measure(positive_label=1),
                "weighted_f1": WeightedF1Measure(),
                "fbeta": FBetaMeasure(average='micro')
        }
        
        weights = loss_weights and torch.FloatTensor(loss_weights)
        self.loss = nn.CrossEntropyLoss(weight=weights)

        initializer(self)

    # @torchsnooper.snoop(watch=('label', ))# @overrides
    def forward(self,  # type: ignore
                tokens: Dict[str, torch.LongTensor],
                label: torch.LongTensor = None,
                ) -> Dict[str, torch.Tensor]:
        """
        Parameters
        ----------
        tokens : Dict[str, Variable], required
            The output of a ListField.
        label : Variable, optional (default = None)
            A variable representing the label for each instance in the batch.

        Returns
        -------
        An output dictionary consisting of:
        class_probabilities : torch.FloatTensor
            A tensor of shape ``(batch_size, num_classes)`` representing a distribution over the
            label classes for each instance.
        loss : torch.FloatTensor, optional
            A scalar loss to be optimised.
        """
        self.saved_inp = tokens
        
        # logger.warn(f"Tokens={tokens['tokens'].dim()}")
        if tokens['tokens'].dim() >= 3:
            word_mask = util.get_text_field_mask(tokens, num_wrapping_dims=1)
            turn_mask = util.get_text_field_mask(tokens)
            args = (word_mask, turn_mask)
        else:
            args = (util.get_text_field_mask(tokens), )
        
        # For char embedding, must provide wrapping dims.
        # https://github.com/allenai/allennlp/issues/820
        e_conv = self.text_field_embedder(tokens, num_wrapping_dims=1)
        # e_conv = self.proj(e_conv)
        
        context, question, knowledge = self.input_unit(e_conv, *args) # , word_mask, turn_mask)
        memory = self.mac(context, question, knowledge)  # tensor of memories
        out = torch.cat([memory, question], 1)

        logits = self.classifier(out)
        
        self.saved_y = logits

        output_dict = {
            'logits': logits
        }
        if label is not None:
            output_dict["loss"] = self.loss(logits, label)
            
            for metric in self.metrics.values():
                metric(logits, label)

        return output_dict

    @overrides
    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        return {
            # f1 get_metric returns (precision, recall, f1)
            "f1": self.metrics["f1"].get_metric(reset=reset)[2],
            "weighted_f1": self.metrics["weighted_f1"].get_metric(reset=reset),
            # "fbeta": self.metrics["fbeta"].get_metric(reset=reset)['fscore'],
            "accuracy": self.metrics["accuracy"].get_metric(reset=reset)
        }