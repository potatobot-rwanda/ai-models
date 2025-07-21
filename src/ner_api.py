from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
import logging

from llm_ner.llm_ner import LLMNER, init_logging

init_logging()

class NerRequest(BaseModel):
    user_message: str
    chat_history: List[str]

app = FastAPI()

ner_models = LLMNER()

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

