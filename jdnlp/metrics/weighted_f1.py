from typing import *
from overrides import overrides

import torch

from allennlp.training.metrics import Metric
from sklearn.metrics import f1_score

@Metric.register('weighted_f1')
class WeightedF1Measure(Metric):
    def __init__(self):
        super().__init__()
        
        self.preds = []
        self.gold_labels = []
    
    def __call__(
        self, predictions: torch.Tensor, 
        gold_labels: torch.Tensor, 
        mask: Optional[torch.Tensor] = None
    ):
        predictions, gold_labels, mask = self.unwrap_to_tensors(predictions, gold_labels, mask)
        predictions = predictions.max(dim=-1)[1]
        
        self.preds.extend(predictions.numpy())
        self.gold_labels.extend(gold_labels.numpy())
        
    @overrides
    def get_metric(self, reset: bool = False):
        """
        Returns
        -------
        A tuple of the following metrics based on the accumulated count statistics:
        precisions : List[float]
        recalls : List[float]
        f1-measures : List[float]
        If ``self.average`` is not ``None``, you will get ``float`` instead of ``List[float]``.
        """
        
        f1 = f1_score(self.preds, self.gold_labels, average='weighted')
        if reset:
            self.reset()
        return f1
    
    @overrides
    def reset(self) -> None:
        self.preds = []
        self.gold_labels = []

        
    
        