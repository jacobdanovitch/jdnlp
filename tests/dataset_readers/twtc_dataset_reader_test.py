from allennlp.common.testing import AllenNlpTestCase
from allennlp.common.util import ensure_list

from jdnlp.dataset_readers import TWTCDatasetReader

class TestTWTCDatasetReader(AllenNlpTestCase):
    def test_read_from_file(self):
        reader = TWTCDatasetReader()
        instances = ensure_list(reader.read('tests/fixtures/twtc_sample.csv'))
        print(instances)
    
    def test_sentence_batching(self):
        reader = TWTCDatasetReader()
        instances = reader.read('tests/fixtures/twtc_sample.csv')

        # wd_per_sent = vars(instances[0].fields["word_per_sentence"])['metadata']
        # sent_per_doc = vars(instances[0].fields["sentence_per_document"])['metadata']

        # assert isinstance(sent_per_doc, int), type(sent_per_doc)
        # assert isinstance(wd_per_sent, list) and set(map(type, wd_per_sent)) == {int}