"""
NER based on LLMs.

Run like this:

python -m llm_ner.llm_ner
"""

import os
import json
import concurrent.futures
from typing import List, Any, Tuple
import logging

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.language_models.chat_models import BaseChatModel

from langchain_community.cache import SQLiteCache
from langchain.globals import set_llm_cache
set_llm_cache(SQLiteCache(database_path=".langchain.db"))

from llm_ner.util import CustomCallback, load_openai_api_key, init_logging

# ner model for a single named entity class
class NERModel:

    def __init__(self, llm, entity_class : str, prompt : str):
        self.llm = llm
        self.entity_class : str = entity_class 
        self.prompt = prompt
        self.nlu_chain = PromptTemplate.from_template(prompt) | self.llm | StrOutputParser()

    def detect(self, user_message, chat_history : List[str]):

        response_callback = CustomCallback()
        response = self.nlu_chain.invoke(
            {
                "user_message": user_message, 
                "chat_history": "\n".join(chat_history),
            },
            {
                "callbacks": [response_callback], 
                "stop_sequences": ["\n"]
            },
        )

        response = response.strip()

        logging.info(f"nlu \"{self.entity_class}\" got response \"{response}\" on message \"{user_message}\"")

        if response.lower() == "none" or response.lower() == "none.":
            result = None
        else:
            result = response

        if result is not None:
            logging.info(f"nlu detected slot \"{self.entity_class}={result}\" in message \"{user_message}\"")

        log_message = {
            "entity_class": self.entity_class,
            "llm_details": {
                key: value for key, value in response_callback.messages.items()
            },
            "result": result
        }
        logging.info(log_message)

        # construct response
        if result is None:
            return None

        start_index = user_message.find(result)
        end_index = start_index * len(result)

        return {
            "entity_class": self.entity_class,
            "surface_value": result,
            "start_index": start_index,
            "end_index": end_index
        }

# instantiate and query the indiviudal NER models together
class LLMNER:

    def __init__(self, llm : BaseChatModel, entity_types = ["last_spray_date", "location", "potato_variety"]):

        self.llm = llm
        
        self.models = []
        for entity_type in entity_types:
            prompt = open(f"llm_ner/prompts/nlu_{entity_type}.txt").read()
            model = NERModel(self.llm, entity_type, prompt)
            self.models.append(model)

    # execute all ners in parallel to reduce waiting time
    def ner(self, user_message, chat_history):

        # helper function for run_ner_parallel: execute a single ner model
        def run_ner(params):
            model, user_message, chat_history = params
            return model.detect(user_message, chat_history)

        with concurrent.futures.ThreadPoolExecutor(max_workers=len(self.models)) as executor:
            params = [(model, user_message, chat_history) for model in self.models]
            results = executor.map(run_ner, params)
            results = list(filter(lambda x:x is not None, results))
            return results

# run this script to run llm ner directly
if __name__ == "__main__":

    init_logging()

    test_data = json.load(open("../data/english/ner/ner.json"))[0:1]

    api_key = load_openai_api_key()
    ner = LLMNER(api_key)
    for sample in test_data:
        
        print(sample)
        chat_history = ["Chatbot: " + sample["preceeding_sentence"]]
        results = ner.ner(sample["input_sentence"], chat_history) 
        print(results)

