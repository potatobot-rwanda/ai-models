import unittest
from llm_ner.llm_ner import LLMNER 

from langchain_community.cache import SQLiteCache
from langchain.globals import set_llm_cache
from langchain_openai.chat_models.base import ChatOpenAI
from llm_ner.util import load_openai_api_key

import logging
logging.basicConfig(level=logging.ERROR)

# set_llm_cache(SQLiteCache(database_path=".langchain.db"))

# Unit test class
class TestNER(unittest.TestCase):

    @unittest.skip("This test costs money and should not be executed often.")
    def test_ner_with_openai(self):

        llm = ChatOpenAI(
            model="gpt-4o",
            openai_api_key=load_openai_api_key(),
        )

        ner_models = LLMNER(llm)
        
        test_sentences = {
            "I sprayed my potatoes last week.": {'entity_class': 'last_spray_date', 'surface_value': 'Last week', 'start_index': -1, 'end_index': -9},
            "My field is located in Musanze.": {'entity_class': 'location', 'surface_value': 'Musanze', 'start_index': 23, 'end_index': 161},
            "I plant potato variety Ndamira.":{'entity_class': 'potato_variety', 'surface_value': 'Ndamira', 'start_index': 23, 'end_index': 161}
        }
        
        for input, expected_response in test_sentences.items():
            response = ner_models.ner(input, [])
            self.assertEqual(response[0], expected_response)
