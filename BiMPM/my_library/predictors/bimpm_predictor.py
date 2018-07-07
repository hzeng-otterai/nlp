from typing import Tuple

from overrides import overrides

from allennlp.common.util import JsonDict
from allennlp.data import Instance
from allennlp.predictors.predictor import Predictor

@Predictor.register('bimpm')
class BiMPMPredictor(Predictor):
    """"Predictor wrapper for the BiMPM"""
    @overrides
    def _json_to_instance(self, json_dict: JsonDict) -> Tuple[Instance, JsonDict]:

        instance = self._dataset_reader.text_to_instance(s1=json_dict['s1'], s2=json_dict['s2'])

        label_dict = self._model.vocab.get_index_to_token_vocabulary('label')
        all_labels = [label_dict[i] for i in range(len(label_dict))]

        return instance, {"all_labels": all_labels}
