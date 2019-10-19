import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy

from typing import Dict, Optional
from overrides import overrides
from copy import deepcopy
import logging

from allennlp.common.checks import ConfigurationError
from allennlp.data import Vocabulary
from allennlp.modules import FeedForward, Seq2VecEncoder, TextFieldEmbedder
from allennlp.models.model import Model
from allennlp.nn import InitializerApplicator, RegularizerApplicator
from allennlp.nn import util, Activation
from allennlp.training.metrics import CategoricalAccuracy, F1Measure, BooleanAccuracy

from jdnlp.modules.loss_functions.contrastive import ContrastiveLoss

logger = logging.getLogger(__name__)

@Model.register("siamese")
class SiameseNetwork(Model):
    """
    This ``Model`` performs text classification for an academic paper.  We assume we're given a
    title and an abstract, and we predict some output label.

    The basic model structure: we'll embed the title and the abstract, and encode each of them with
    separate Seq2VecEncoders, getting a single vector representing the content of each.  We'll then
    concatenate those two vectors, and pass the result through a feedforward network, the output of
    which we'll use as our scores for each label.

    Parameters
    ----------
    vocab : ``Vocabulary``, required
        A Vocabulary, required in order to compute sizes for input/output projections.
    text_field_embedder : ``TextFieldEmbedder``, required
        Used to embed the ``tokens`` ``TextField`` we get as input to the model.
    encoder : ``Seq2VecEncoder``
        The encoder that we will use to convert the title to a vector.
    classifier_feedforward : ``FeedForward``
    initializer : ``InitializerApplicator``, optional (default=``InitializerApplicator()``)
        Used to initialize the model parameters.
    regularizer : ``RegularizerApplicator``, optional (default=``None``)
        If provided, will be used to calculate the regularization penalty during training.
    """
    def __init__(self, vocab: Vocabulary,
                 text_field_embedder: TextFieldEmbedder,
                 left_encoder: Seq2VecEncoder,
                 right_encoder: Seq2VecEncoder = None,
                 loss_margin: float = 1.0,
                 prediction_threshold: float = 0.8,
                 initializer: InitializerApplicator = InitializerApplicator(),
                 regularizer: Optional[RegularizerApplicator] = None) -> None:
        super().__init__(vocab, regularizer)

        self.prediction_threshold = prediction_threshold

        self.text_field_embedder = text_field_embedder
        
        self.left_encoder = left_encoder
        self.right_encoder = right_encoder or deepcopy(left_encoder)

        if text_field_embedder.get_output_dim() != self.left_encoder.get_input_dim():
            raise ConfigurationError("The output dimension of the text_field_embedder must match the "
                                     "input dimension of the encoder. Found {} and {}, "
                                     "respectively.".format(text_field_embedder.get_output_dim(),
                                                            self.left_encoder.get_input_dim()))
        self.metrics = {
                "accuracy": BooleanAccuracy()
                # "accuracy": CategoricalAccuracy(),
                # "f1": F1Measure(positive_label=1)
        }
        self.loss = ContrastiveLoss(loss_margin)
        initializer(self)

    @overrides
    def forward(self,  # type: ignore
                left: Dict[str, torch.LongTensor],
                right: Dict[str, torch.Tensor],
                label: torch.LongTensor = None) -> Dict[str, torch.Tensor]:
        """
        Parameters
        ----------
        tokens : Dict[str, Variable], required
            The output of ``TextField.as_array()``.
        sentence_per_document : Dict[str, torch.Tensor], required
            The number of sentences for each document.
        word_per_sentence : Dict[str, torch.Tensor], required
            The number of words for each sentence in each document.
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

        left_embedded = self.text_field_embedder(left)
        left_mask = util.get_text_field_mask(left)

        right_embedded = self.text_field_embedder(right)
        right_mask = util.get_text_field_mask(right)

        v_l = self.left_encoder(left_embedded, left_mask)
        v_r = self.right_encoder(right_embedded, right_mask)

        loss = self.loss(v_l, v_r, label)
        sim = F.cosine_similarity(v_l, v_r) > self.prediction_threshold

        output_dict = {
            'loss': loss
        }

        for metric in self.metrics.values():
            #logging.info(f"Sim {sim}")
            #logging.info(f"Label {label}")
            metric(sim.long(), label.long())

        return output_dict

    @overrides
    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        return {metric_name: metric.get_metric(reset) for metric_name, metric in self.metrics.items()}

