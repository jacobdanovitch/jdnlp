"""
Literally copied from here:
https://github.com/allenai/allennlp/blob/master/allennlp/models/basic_classifier.py

Adding ability to specify basic metrics. Surely there must be a better way than this.
"""

from typing import Dict, Optional, List
from overrides import overrides

import torch

from allennlp.models.model import Model
from allennlp.training.metrics import CategoricalAccuracy, F1Measure
from allennlp.training.metrics.metric import Metric

from allennlp.data import Vocabulary
from allennlp.models.model import Model
from allennlp.modules import Seq2SeqEncoder, Seq2VecEncoder, TextFieldEmbedder
from allennlp.nn import InitializerApplicator, RegularizerApplicator
from allennlp.nn.util import get_text_field_mask
from allennlp.training.metrics import CategoricalAccuracy

@Model.register("classifier_with_metrics")
class ClassifierWithMetrics(Model):
    def __init__(
        self,
        metrics: Dict[str, Dict[str, any]],
        vocab: Vocabulary,
        text_field_embedder: TextFieldEmbedder,
        seq2vec_encoder: Seq2VecEncoder,
        seq2seq_encoder: Seq2SeqEncoder = None,
        dropout: float = None,
        num_labels: int = None,
        label_namespace: str = "labels",
        initializer: InitializerApplicator = InitializerApplicator(),
        regularizer: Optional[RegularizerApplicator] = None,
        loss_weights: Optional[List] = None
    ) -> None:

        super().__init__(vocab, regularizer)
        
        self.metrics = {m: Metric.by_name(m)(**kwargs) for (m, kwargs) in metrics.items()}
        print(metrics.items())
        print(self.metrics)
        
        self._text_field_embedder = text_field_embedder

        if seq2seq_encoder:
            self._seq2seq_encoder = seq2seq_encoder
        else:
            self._seq2seq_encoder = None

        self._seq2vec_encoder = seq2vec_encoder
        self._classifier_input_dim = self._seq2vec_encoder.get_output_dim()

        if dropout:
            self._dropout = torch.nn.Dropout(dropout)
        else:
            self._dropout = None

        self._label_namespace = label_namespace

        if num_labels:
            self._num_labels = num_labels
        else:
            self._num_labels = vocab.get_vocab_size(namespace=self._label_namespace)
        self._classification_layer = torch.nn.Linear(self._classifier_input_dim, self._num_labels)
        self._accuracy = CategoricalAccuracy()
        
        Loss = torch.nn.CrossEntropyLoss
        self._loss = Loss(weight=torch.tensor(loss_weights)) if loss_weights else Loss()
        
        initializer(self)

    def forward(  # type: ignore
        self, tokens: Dict[str, torch.LongTensor], label: torch.IntTensor = None
    ) -> Dict[str, torch.Tensor]:

        """
        Parameters
        ----------
        tokens : Dict[str, torch.LongTensor]
            From a ``TextField``
        label : torch.IntTensor, optional (default = None)
            From a ``LabelField``
        Returns
        -------
        An output dictionary consisting of:
        logits : torch.FloatTensor
            A tensor of shape ``(batch_size, num_labels)`` representing
            unnormalized log probabilities of the label.
        probs : torch.FloatTensor
            A tensor of shape ``(batch_size, num_labels)`` representing
            probabilities of the label.
        loss : torch.FloatTensor, optional
            A scalar loss to be optimised.
        """
        embedded_text = self._text_field_embedder(tokens)
        mask = get_text_field_mask(tokens).float()

        if self._seq2seq_encoder:
            embedded_text = self._seq2seq_encoder(embedded_text, mask=mask)

        embedded_text = self._seq2vec_encoder(embedded_text, mask=mask)

        if self._dropout:
            embedded_text = self._dropout(embedded_text)

        logits = self._classification_layer(embedded_text)
        probs = torch.nn.functional.softmax(logits, dim=-1)

        output_dict = {"logits": logits, "probs": probs}

        if label is not None:
            loss = self._loss(logits, label.long().view(-1))
            output_dict["loss"] = loss
            for metric in self.metrics.values():
                metric(logits, label)
            self._accuracy(logits, label)

        return output_dict

    @overrides
    def decode(self, output_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Does a simple argmax over the probabilities, converts index to string label, and
        add ``"label"`` key to the dictionary with the result.
        """
        predictions = output_dict["probs"]
        if predictions.dim() == 2:
            predictions_list = [predictions[i] for i in range(predictions.shape[0])]
        else:
            predictions_list = [predictions]
        classes = []
        for prediction in predictions_list:
            label_idx = prediction.argmax(dim=-1).item()
            label_str = self.vocab.get_index_to_token_vocabulary(self._label_namespace).get(
                label_idx, str(label_idx)
            )
            classes.append(label_str)
        output_dict["label"] = classes
        return output_dict
    
    
    """
    Doesn't work with f1 for some reason.
    "f1": {
                "positive_label": 1
            }
    Throws ridiculous RuntimeError("You never call this metric before.")
    """
    @overrides
    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        return {name: parse_metric(metric, reset) for (name, metric) in self.metrics.items()}
    
    
    
def parse_metric(metric, reset):
    m = metric.get_metric(reset=reset)
    if isinstance(m, list) or isinstance(m, tuple): 
        return m[-1]
    
    return m