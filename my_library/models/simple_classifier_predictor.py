from typing import Tuple

from overrides import overrides

from allennlp.common.util import JsonDict
from allennlp.data import Instance
from allennlp.predictors.predictor import Predictor

@Predictor.register('simple_classifier')
class SimpleClassifierPredictor(Predictor):

    def predict(self, text: str) -> JsonDict:
        return self.predict_json({"text" : text})
        
    @overrides
    def _json_to_instance(self, json_dict: JsonDict) -> Instance:
        text = json_dict['text']
        instance = self._dataset_reader.text_to_instance(text=text)

        return instance