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
    def _json_to_instance(self, json_dict: JsonDict) -> Tuple[Instance, JsonDict]:
        instance = self._dataset_reader.text_to_instance(s1=json_dict['s1'], s2=json_dict['s2'])

        label_dict = self._model.vocab.get_index_to_token_vocabulary('labels')
        all_labels = [label_dict[i] for i in range(len(label_dict))]

        return instance, {"all_labels": all_labels}