from typing import Tuple

from overrides import overrides

from allennlp.common.util import JsonDict
from allennlp.data import Instance
from allennlp.predictors.predictor import Predictor

@Predictor.register('sentence_pair')
class SentencePairPredictor(Predictor):
    """"Predictor wrapper for the sentence pairs"""
    def predict(self, premise: str, hypothesis: str) -> JsonDict:
        return self.predict_json({"premise": premise, "hypothesis": hypothesis})

    @overrides
    def _json_to_instance(self, json_dict: JsonDict) -> Instance:
        return self._dataset_reader.text_to_instance(premise=json_dict['premise'], hypothesis=json_dict['hypothesis'])
