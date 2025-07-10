from entities.data import ServiceCheckResult, BotMessage
from use_cases.ports.ml_service import IMLServiceRepository
import pickle

"""
SERVICE FOR DETECTING AD & COMPETITOR INFORMATION

PIPELINE:
1. Detection of the message's language (Russian / English)
2. Deletion of links, HTML-inserts and some special symbols
3. Tokenization of the text
4. Running rule engine (cheching 3 categories: promo / competiton / common text)
5. Running TF-IDF + linear classification (cheching 3 categories)
"""


class AdFilterRepository(IMLServiceRepository):
    def __init__(self, model_path: str):
        self.filename = model_path
        with open(self.filename, "rb") as file:
            self.model = pickle.load(file)

    def process(self, message: BotMessage) -> ServiceCheckResult:
        text = message.answer or ""
        proba = float(self.model.predict_proba([text])[0, 1])
        label = 0 if proba >= 0.5 else 1
        return ServiceCheckResult(safe=label, score=proba, masked_answer=message.answer, question=message.question)
