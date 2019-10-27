from allennlp.common.testing import ModelTestCase

class MAC_KB_Test(ModelTestCase):
    def setUp(self):
        super().setUp()
        self.set_up_model('tests/fixtures/kb_mac.jsonnet', 'datasets/counterspeech/sample.json')

    def test_model_can_train_save_and_load(self):
        self.ensure_model_can_train_save_and_load(self.param_file)