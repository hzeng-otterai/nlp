# pylint: disable=no-self-use,invalid-name
from allennlp.common.testing import AllenNlpTestCase
from allennlp.common.util import ensure_list

from hznlp.dataset_readers import QuoraParaphraseDatasetReader

class TestQuoraParaphraseDatasetReader(AllenNlpTestCase):
    def test_read_from_file(self):

        reader = QuoraParaphraseDatasetReader()
        instances = ensure_list(reader.read('tests/quora_train_sample.tsv'))

        assert len(instances) == 10
        for x in instances:
            for f in ["label", "s1", "s2"]:
                assert f in x.fields
