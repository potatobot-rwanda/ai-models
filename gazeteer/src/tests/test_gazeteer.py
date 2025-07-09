"""
Call the unit test:

python -m unittest tests/test_gazeteer.py
"""

import unittest
from gazeteer import EngPotatoGazeteer

class TestPotatoDetector(unittest.TestCase):

    def test_potato_detector(self):
        data = [
            {
                "sentence": "I plant red bliss potatoes.",
                "entities": [
                {
                    "normalized_value": "Red Bliss",
                    "start_index": 8,
                    "end_index": 17,
                    "surface_value": "red bliss",
                    "annotation_type": "POTATO"
                }
                ]
            },
            {
                "sentence": "Yukon gold is perfect for mashing.",
                "entities": [
                {
                    "normalized_value": "Yukon Gold",
                    "start_index": 0,
                    "end_index": 10,
                    "surface_value": "Yukon gold",
                    "annotation_type": "POTATO"
                }
                ]
            },
            {
                "sentence": "I roasted some purple majesty last night.",
                "entities": [
                {
                    "normalized_value": "Purple Majesty",
                    "start_index": 18,
                    "end_index": 33,
                    "surface_value": "purple majesty",
                    "annotation_type": "POTATO"
                }
                ]
            },
            {
                "sentence": "The market had fresh la ratte on display.",
                "entities": [
                {
                    "normalized_value": "La Ratte",
                    "start_index": 23,
                    "end_index": 31,
                    "surface_value": "la ratte",
                    "annotation_type": "POTATO"
                }
                ]
            },
            {
                "sentence": "I prefer German butterball for potato salad.",
                "entities": [
                {
                    "normalized_value": "German Butterball",
                    "start_index": 9,
                    "end_index": 27,
                    "surface_value": "German butterball",
                    "annotation_type": "POTATO"
                }
                ]
            },
            {
                "sentence": "I planted Yukon gold and red bliss this season.",
                "entities": [
                {
                    "normalized_value": "Yukon Gold",
                    "start_index": 10,
                    "end_index": 20,
                    "surface_value": "Yukon gold",
                    "annotation_type": "POTATO"
                },
                {
                    "normalized_value": "Red Bliss",
                    "start_index": 25,
                    "end_index": 34,
                    "surface_value": "red bliss",
                    "annotation_type": "POTATO"
                }
                ]
            },
            {
                "sentence": "Maris piper, king edward, and desiree are common in the UK.",
                "entities": [
                {
                    "normalized_value": "Maris Piper",
                    "start_index": 0,
                    "end_index": 11,
                    "surface_value": "Maris piper",
                    "annotation_type": "POTATO"
                },
                {
                    "normalized_value": "King Edward",
                    "start_index": 13,
                    "end_index": 24,
                    "surface_value": "king edward",
                    "annotation_type": "POTATO"
                },
                {
                    "normalized_value": "Desiree",
                    "start_index": 30,
                    "end_index": 38,
                    "surface_value": "desiree",
                    "annotation_type": "POTATO"
                }
                ]
            },
            {
                "sentence": "I cooked la ratte with purple majesty and rose finn apple.",
                "entities": [
                {
                    "normalized_value": "La Ratte",
                    "start_index": 9,
                    "end_index": 17,
                    "surface_value": "la ratte",
                    "annotation_type": "POTATO"
                },
                {
                    "normalized_value": "Purple Majesty",
                    "start_index": 23,
                    "end_index": 38,
                    "surface_value": "purple majesty",
                    "annotation_type": "POTATO"
                },
                {
                    "normalized_value": "Rose Finn Apple",
                    "start_index": 43,
                    "end_index": 59,
                    "surface_value": "rose finn apple",
                    "annotation_type": "POTATO"
                }
                ]
            }

        ]

        gazeteer = EngPotatoGazeteer()
        for item in data:
            sentence = item["sentence"]
            entities = item["entities"]
            matches = gazeteer.match(sentence, case_sensitive=False)
            self.assertEqual(len(matches), len(entities), f"Mismatch in number of entities for sentence: {sentence}")

            for entity in entities:
                found = False
                for match in matches:
                    if (match["start_index"] == entity["start_index"] and
                        match["end_index"] == entity["end_index"] and
                        match["normalized_value"] == entity["normalized_value"]):
                        found = True
                        break
                self.assertTrue(found, f"Entity not found: {entity} in sentence: {sentence}")


if __name__ == '__main__':
    unittest.main()
    
