import os
import getpass
import logging
from logging.handlers import RotatingFileHandler
import json
import concurrent.futures
from typing import List, Any, Tuple

from langchain_core.output_parsers import StrOutputParser
from langchain.callbacks.base import BaseCallbackHandler
from langchain_core.prompts import PromptTemplate
from langchain_core.outputs import LLMResult
from langchain_openai import ChatOpenAI

from langchain_community.cache import SQLiteCache
from langchain.globals import set_llm_cache
set_llm_cache(SQLiteCache(database_path=".langchain.db"))

API_KEY = os.getenv("OPENAI_API_KEY")
if not API_KEY:
    API_KEY = getpass.getpass("Enter your OPENAI_API_KEY: ")


def init_logging(console_level = logging.WARNING, file_level = logging.INFO):
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    formatter = logging.Formatter(
        "%(asctime)s - %(filename)s:%(lineno)d - %(levelname)s - %(message)s"
    )
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    console_handler.setLevel(console_level)
    logger.addHandler(console_handler)

    file_handler = RotatingFileHandler('app.log', maxBytes=1024 * 1024, backupCount=5)
    file_handler.setFormatter(formatter)
    file_handler.setLevel(file_level)
    logger.addHandler(file_handler)


def init_llm():
    llm = ChatOpenAI(
        model="gpt-4o",
        openai_api_key=API_KEY,
    )
    return llm

# https://python.langchain.com/v0.1/docs/modules/callbacks/
# custom langchain callback to log llm details
class CustomCallback(BaseCallbackHandler):

    def __init__(self):
        self.messages = {}

    def on_llm_start(
        self, serialized: dict[str, Any], prompts: list[str], **kwargs: Any
    ) -> Any:
        self.messages["on_llm_start_prompts"] = prompts
        self.messages["on_llm_start_kwargs"] = kwargs

    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> Any:

        llm_generation = []
        for gen in response.generations:
            for gen2 in gen:
                llm_generation.append({
                    "text": gen2.text,
                    "generation_info": gen2.generation_info
                })

        self.messages["on_llm_end_response"] = llm_generation
        self.messages["on_llm_end_kwargs"] = kwargs

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

# execute all ners in parallel to reduce waiting time
def run_ner_parallel(models, user_message, chat_history):

    # helper function for run_ner_parallel: execute a single ner model
    def run_ner(params):
        model, user_message, chat_history = params
        return model.detect(user_message, chat_history)

    with concurrent.futures.ThreadPoolExecutor(max_workers=len(models)) as executor:
        params = [(model, user_message, chat_history) for model in models]
        results = executor.map(run_ner, params)
        results = list(filter(lambda x:x is not None, results))
        return results

# run this script to run llm ner directly
if __name__ == "__main__":

    init_logging()

    entity_types = ["last_spray_date", "location", "plant_date", "potato_variety"]
    llm = init_llm()
    models = []
    for entity_type in entity_types:
        prompt = open(f"prompts/nlu_{entity_type}.txt").read()
        model = NERModel(llm, entity_type, prompt)
        models.append(model)

    test_data = json.load(open("../../data/english/ner/ner.json"))[0:1]
    for sample in test_data:
        
        print(sample)
        chat_history = ["Chatbot: " + sample["preceeding_sentence"]]
        results = run_ner_parallel(models, sample["input_sentence"], chat_history) 
        print(results)
