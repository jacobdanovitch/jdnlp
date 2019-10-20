from allennlp.common.testing import AllenNlpTestCase
from allennlp.common.util import ensure_list

from jdnlp.dataset_readers import MultiTurnReader

class TestMultiTurnReader(AllenNlpTestCase):
    def test_read_from_file(self):
        reader = MultiTurnReader(use_cache=False)
        instances = ensure_list(reader.read('tests/fixtures/multiturn_sample.json'))
        print(instances)