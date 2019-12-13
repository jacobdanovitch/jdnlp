import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy

from typing import Dict, Optional
from overrides import overrides
from copy import deepcopy

from allennlp.common.checks import ConfigurationError
from allennlp.data import Vocabulary
from allennlp.modules import FeedForward, Seq2VecEncoder, TextFieldEmbedder
from allennlp.models.model import Model
from allennlp.nn import InitializerApplicator, RegularizerApplicator
from allennlp.nn import util, Activation
from allennlp.training.metrics import CategoricalAccuracy, F1Measure, BooleanAccuracy

from jdnlp.modules.loss_functions.triplet import TripletLoss
from pytorch_metric_learning import losses, miners

import torchsnooper
import logging
logger = logging.getLogger(__name__)

@Model.register("siamese_triplet_loss")
class SiameseNetworkTripletLoss(Model):
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
    initializer : ``InitializerApplicator``, optional (default=``InitializerApplicator()``)
        Used to initialize the model parameters.
    regularizer : ``RegularizerApplicator``, optional (default=``None``)
        If provided, will be used to calculate the regularization penalty during training.
    """
    def __init__(self, vocab: Vocabulary,
                 text_field_embedder: TextFieldEmbedder,
                 encoder: Seq2VecEncoder,
                 right_encoder: Seq2VecEncoder = None,
                 loss_margin: float = 1.0,
                 return_embeddings: bool = False,
                 initializer: InitializerApplicator = InitializerApplicator(),
                 regularizer: Optional[RegularizerApplicator] = None) -> None:
        super().__init__(vocab, regularizer)

        self.text_field_embedder = text_field_embedder
        
        self.encoder = encoder
        self.right_encoder = right_encoder or self.encoder # deepcopy()
        
        self.return_embeddings = return_embeddings

        if text_field_embedder.get_output_dim() != self.encoder.get_input_dim():
            raise ConfigurationError("The output dimension of the text_field_embedder must match the "
                                     "input dimension of the encoder. Found {} and {}, "
                                     "respectively.".format(text_field_embedder.get_output_dim(),
                                                            self.encoder.get_input_dim()))
        self.metrics = {
                "accuracy": BooleanAccuracy()
                # "accuracy": CategoricalAccuracy(),
                # "f1": F1Measure(positive_label=1)
        }
        
        self.miner = miners.MultiSimilarityMiner(epsilon=0.1)
        
        # """
        self.loss = losses.TripletMarginLoss(
            margin=loss_margin,
            triplets_per_anchor="all",
             # distance_norm=1,
             # normalize_embeddings=False,
             # swap=True,
             # smooth_loss=True
        ) 
        # """
        # self.loss = losses.MultiSimilarityLoss(alpha=0.1, beta=40)
        #self.loss = losses.ContrastiveLoss()
        # TripletLoss(loss_margin)

        initializer(self)

    # @torchsnooper.snoop()
    def embed_and_mask_tokens(self, tokens):
        # num_wrapping_dims = 1 if ('token_characters' in tokens) or (tokens['tokens'].dim() >= 3) else 0
        num_wrapping_dims = len(tokens.keys()) - 1
        if num_wrapping_dims == 1:
            word_mask = util.get_text_field_mask(tokens, num_wrapping_dims=1)
            sentence_mask = util.get_text_field_mask(tokens)
            
            masks = (word_mask, sentence_mask)
        else:
            masks = (util.get_text_field_mask(tokens), )
        
        e = self.text_field_embedder(tokens, num_wrapping_dims=num_wrapping_dims)
        return e, masks

    # @torchsnooper.snoop(watch=('loss', 'self.loss.num_non_zero_triplets'))#@overrides
    def forward(self,  # type: ignore
                anchor: Dict[str, torch.LongTensor],
                positive: Dict[str, torch.LongTensor]=None,
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

        anchor_embedded, anchor_masks = self.embed_and_mask_tokens(anchor)
        v_a = self.encoder(anchor_embedded, *anchor_masks)
        
        if positive is None:
            return {'embedding': v_a}
        
        pos_embedded, pos_masks = self.embed_and_mask_tokens(positive)
        v_p = self.right_encoder(pos_embedded, *pos_masks)   
        
        # logger.warn(v_a)
        output_dict = {'pos_similarity': F.cosine_similarity(v_a, v_p)}
        if label is not None:
            embeddings = torch.cat([v_a, v_p], dim=0)
            labels = torch.cat([label, label], dim=0)
             
            hard_pairs = self.miner(embeddings, labels)
            loss = self.loss(embeddings, labels, hard_pairs) # * 1000
            loss = loss or (embeddings * 0).sum()
            output_dict['loss'] = loss
        
        if self.return_embeddings:
            output_dict.update({
                'anchor': v_a,
                'positive': v_p
            })

        """
        for metric in self.metrics.values():
            pred = torch.tensor(pos_sim > neg_sim).long()
            # print(pred.size())
            # print(pred)
            metric(pred, torch.tensor([1]*pred.size(0)).long())
        """

        return output_dict

    @overrides
    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        return {metric_name: metric.get_metric(reset) for metric_name, metric in self.metrics.items()}

