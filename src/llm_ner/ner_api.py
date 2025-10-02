from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
import logging
import os

from langchain_openai.chat_models.base import ChatOpenAI
from llm_ner.llm_ner import LLMNER, init_logging
from llm_ner.util import load_openai_api_key

init_logging()

class NerRequest(BaseModel):
    user_message: str
    chat_history: List[str]

app = FastAPI()
ner_models = None

def init_ner(llm = None):
    global ner_models
    if llm is None:
        llm = ChatOpenAI(
            model="gpt-4o",
            openai_api_key=load_openai_api_key(),
        )
    ner_models = LLMNER(llm)

# we add this switch so we can initialize the NER for unit testing with a mock llm
if os.getenv('ENV', 'production') == 'production':
    init_ner()

@app.post("/api/ner")
async def ner_request(nerRequest : NerRequest):

    global ner_models
    try:
        results = ner_models.ner(nerRequest.user_message, nerRequest.chat_history)
        return {
            "user_message": nerRequest.user_message,
            "chat_history": nerRequest.chat_history,
            "ner_results": results
        }
    except Exception as e:
        logging.exception(e)

