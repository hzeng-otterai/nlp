# pylint: disable=invalid-name,protected-access
from allennlp.common.testing import ModelTestCase


class BiMPMTest(ModelTestCase):
    def setUp(self):
        super(BiMPMTest, self).setUp()
        self.set_up_model('tests/quora.json',
                          'tests/quora_sample.tsv')

    def test_model_can_train_save_and_load(self):
        self.ensure_model_can_train_save_and_load(self.param_file)
