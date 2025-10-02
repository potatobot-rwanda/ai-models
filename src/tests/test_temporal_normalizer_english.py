import unittest
from llm_ner.temporal_normalizer import TemporalNormalizerEnglish

import logging
from datetime import datetime

logging.basicConfig(level=logging.ERROR)

class TestTemporalNormalizer(unittest.TestCase):

    def test_temporal_analyer(self):

        ta = TemporalNormalizerEnglish()

        result = ta.analyze('3 week ago', reference_date=datetime(2023, 10, 10))
        self.assertEqual(result, datetime(2023, 9, 19))

        result = ta.analyze('June 1st', reference_date=datetime(2023, 10, 10))
        self.assertEqual(result, datetime(2023, 6, 1))

