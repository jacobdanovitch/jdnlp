from copy import deepcopy
from typing import List, Dict

from overrides import overrides
import numpy

from allennlp.common.util import JsonDict
from allennlp.data import Instance
from allennlp.predictors.predictor import Predictor
from allennlp.data.fields import LabelField
from allennlp.data import DatasetReader, Instance
from allennlp.models import Model


@Predictor.register("knn_classifier")
class KNNClassifierPredictor(Predictor):
    """
    
    """
    def __init__(
        self, 
        model: Model, 
        dataset_reader: DatasetReader
    ) -> None:
        super().__init__(model, dataset_reader)
        model.return_embeddings = True

    def predict(self, sentence: str) -> JsonDict:
        return self.predict_json({"sentence": sentence})

    
    @overrides
    def _json_to_instance(self, json_dict: JsonDict) -> Instance:
        """
        Expects JSON that looks like ``{"sentence": "..."}``.
        """
        if 'text_idx' in json_dict:
            x = (json_dict['text_idx'], json_dict['comment_idx'])# , json_dict['comment_idx'])
            return self._dataset_reader.text_to_instance(*x)  # type: ignore
        return self._dataset_reader.text_to_instance(*json_dict.values())

    @overrides
    def predictions_to_labeled_instances(
        self, instance: Instance, outputs: Dict[str, numpy.ndarray]
    ) -> List[Instance]:
        new_instance = deepcopy(instance)
        label = numpy.argmax(outputs["probs"])
        new_instance.add_field("label", LabelField(int(label), skip_indexing=True))
        return [new_instance]