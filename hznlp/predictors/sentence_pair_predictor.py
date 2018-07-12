from typing import Tuple

from overrides import overrides

from allennlp.common.util import JsonDict
from allennlp.data import Instance
from allennlp.predictors.predictor import Predictor

@Predictor.register('sentence_pair')
class SentencePairPredictor(Predictor):
    """"Predictor wrapper for the sentence pairs"""
    def predict(self, s1: str, s2: str) -> JsonDict:
        return self.predict_json({"s1": s1, "s2": s2})

    @overrides
    def _json_to_instance(self, json_dict: JsonDict) -> Instance:
        instance = self._dataset_reader.text_to_instance(s1=json_dict['s1'], s2=json_dict['s2'])

        return instance