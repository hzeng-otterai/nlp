# pylint: disable=invalid-name,protected-access
from allennlp.common.testing import ModelTestCase


class BiMPMTest(ModelTestCase):
    def setUp(self):
        super(BiMPMTest, self).setUp()
        self.set_up_model('tests/bimpm_quora_test.json',
                          'tests/quora_train_sample.tsv')

    def test_model_can_train_save_and_load(self):
        self.ensure_model_can_train_save_and_load(self.param_file)


class ParaClassificationTest(ModelTestCase):
    def setUp(self):
        super(ParaClassificationTest, self).setUp()
        self.set_up_model('tests/para_classification_quora_test.json',
                          'tests/quora_train_sample.tsv')

    def test_model_can_train_save_and_load(self):
        self.ensure_model_can_train_save_and_load(self.param_file)
