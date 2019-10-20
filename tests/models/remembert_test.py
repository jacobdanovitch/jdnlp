from allennlp.common.testing import ModelTestCase

class ReMemBERTTest(ModelTestCase):
    def setUp(self):
        super().setUp()
        self.set_up_model('tests/fixtures/remembert.jsonnet', 'tests/fixtures/multiturn_sample.json')

    def test_model_can_train_save_and_load(self):
        self.ensure_model_can_train_save_and_load(self.param_file)