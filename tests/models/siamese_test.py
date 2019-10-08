from allennlp.common.testing import ModelTestCase

class SiameseNetworkTest(ModelTestCase):
    def setUp(self):
        super().setUp()
        self.set_up_model('tests/fixtures/siamese_test.json', 'tests/fixtures/nnm_sample.csv')

    def test_model_can_train_save_and_load(self):
        self.ensure_model_can_train_save_and_load(self.param_file)