from allennlp.common.testing import ModelTestCase

class TwtcClassifierTest(ModelTestCase):
    def setUp(self):
        super().setUp()
        self.set_up_model('tests/fixtures/twtc_classifier.json', 'tests/fixtures/twtc_sample.csv')

    def test_model_can_train_save_and_load(self):
        self.ensure_model_can_train_save_and_load(self.param_file)