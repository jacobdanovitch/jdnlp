from allennlp.common.testing import AllenNlpTestCase
from allennlp.common.util import ensure_list

from jdnlp.data import DataTransform
from jdnlp.data.split import DataSplitter
from jdnlp.commands.transform_data import data_transform_from_file

import pandas as pd

import math
import logging

logger = logging.getLogger(__name__)

class TestDataSplitter(AllenNlpTestCase):
    def test_can_register(self):
        tf = DataTransform.by_name("data_splitter")
        assert isinstance(tf, type(DataSplitter)), type(tf)

        logger.info(tf)

    def test_can_read(self):
        fp = "tests/fixtures/twtc_sample.csv"
        df = pd.read_csv(fp, header=None)

        tf = data_transform_from_file("tests/fixtures/data_transforms/data_split.json")


    def test_can_split(self):
        fp = "tests/fixtures/twtc_sample.csv"
        df = pd.read_csv(fp, header=None)

        tf = DataSplitter(df)
        train, test = tf.transform(test_size=0.2)

        assert len(df) == len(train) + len(test)