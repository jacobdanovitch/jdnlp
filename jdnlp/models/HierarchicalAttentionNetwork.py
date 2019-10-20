import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy

from typing import Dict, Optional
from overrides import overrides

from allennlp.common.checks import ConfigurationError
from allennlp.data import Vocabulary
from allennlp.modules import FeedForward, Seq2VecEncoder, Seq2SeqEncoder, TextFieldEmbedder, Embedding
from allennlp.models.model import Model
from allennlp.nn import InitializerApplicator, RegularizerApplicator
from allennlp.nn import util, Activation
from allennlp.training.metrics import CategoricalAccuracy, F1Measure

import logging
logger = logging.getLogger(__name__)

@Model.register("HAN")
class HierarchialAttentionNetwork(Model):
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
                 text_field_embedder: TextFieldEmbedder,
                 encoder: Seq2VecEncoder,
                 classifier_feedforward: FeedForward = None,
                 loss_weights=[1, 1],
                 initializer: InitializerApplicator = InitializerApplicator(),
                 regularizer: Optional[RegularizerApplicator] = None) -> None:
        super().__init__(vocab, regularizer)

        self.text_field_embedder = text_field_embedder
        self.num_classes = self.vocab.get_vocab_size("labels")
        
        self.encoder = encoder.train()

        hidden_size = encoder.get_output_dim()
        self.classifier_feedforward = classifier_feedforward or FeedForward(
            input_dim = hidden_size,
            num_layers = 3,
            hidden_dims = [
                hidden_size,
                hidden_size,
                self.num_classes
            ],
            activations = [
                Activation.by_name("relu")(),
                Activation.by_name("relu")(),
                Activation.by_name("linear")()
            ],
            dropout = [
                0.2,
                0.2,
                0.0
            ]
        )

        if text_field_embedder.get_output_dim() != encoder.get_input_dim():
            raise ConfigurationError("The output dimension of the text_field_embedder must match the "
                                     "input dimension of the encoder. Found {} and {}, "
                                     "respectively.".format(text_field_embedder.get_output_dim(),
                                                            encoder.get_input_dim()))
        self.metrics = {
                "accuracy": CategoricalAccuracy(),
                "f1": F1Measure(positive_label=1)
        }
        
        weights = torch.FloatTensor(loss_weights)
        self.loss = nn.CrossEntropyLoss(weight=weights)

        self.nn = None
        initializer(self)

    @overrides
    def forward(self,  # type: ignore
                tokens: Dict[str, torch.LongTensor],
                label: torch.LongTensor = None
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
        word_attention: torch.FloatTensor
            The attention weights for each word.
        sentence_attention: torch.FloatTensor
            The attention weights for each sentence.
        loss : torch.FloatTensor, optional
            A scalar loss to be optimised.
        """
        
        word_mask = util.get_text_field_mask(tokens, num_wrapping_dims=1)
        sentence_mask = util.get_text_field_mask(tokens)
        
        embedded = self.text_field_embedder(tokens)
        
        doc_vecs, sent_attention, word_attention = self.encoder(embedded, word_mask, sentence_mask)
        logits = self.classifier_feedforward(doc_vecs)
        logits = F.log_softmax(logits, dim=-1)


        output_dict = {
            'logits': logits,
            'word_attention': word_attention,
            'sentence_attention': sent_attention
        }
        if label is not None:
            loss = self.loss(logits, label)
            for metric in self.metrics.values():
                metric(logits, label)
            output_dict["loss"] = loss

        return output_dict

    @overrides
    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        return {
            # f1 get_metric returns (precision, recall, f1)
            "f1": self.metrics["f1"].get_metric(reset=reset)[2],
            "accuracy": self.metrics["accuracy"].get_metric(reset=reset)
        }

