import torch
import time
import torch.nn.functional as F
from entities.data import ServiceCheckResult, BotMessage
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from use_cases.ports.ml_service import IMLServiceRepository
from lingua import Language, LanguageDetectorBuilder

"""
SAFETY CLASSIFIER MODULE

Purpose:
This module implements a safety classification service for Telegram bot messages to detect harmful, toxic, or disturbing content in user responses. It supports both English and Russian languages.

Models:
- English: "ujjawalsah/bert-toxicity-classifier" — a general-purpose BERT model from HuggingFace used for binary classification (non-toxic vs. toxic).
- Russian: "cointegrated/rubert-tiny-toxicity" — a Russian-language BERT model from DeepPavlov. Supports multilabel classification for various toxicity types (e.g., obscene, insult, threat).

Pipeline Overview:
1. Language Detection — detects whether the message is in English or Russian using `langdetect`.
2. Tokenization — the input text is tokenized using the corresponding tokenizer for each model.
3. Classification:
   - For English: uses softmax on logits to obtain the probability of toxic content.
   - For Russian: uses sigmoid activation to evaluate multiple toxicity labels, then takes the highest probability among toxic labels.
4. Thresholding — if the toxicity score is above 0.1, the message is considered harmful.
"""

detector = LanguageDetectorBuilder.from_languages(
    Language.ENGLISH, Language.RUSSIAN
).build()


def detect_language(text: str) -> str:
    lang = detector.detect_language_of(text)
    if lang == Language.ENGLISH:
        return "en"
    elif lang == Language.RUSSIAN:
        return "ru"
    return "unknown"


EN_MODEL_NAME = "ujjawalsah/bert-toxicity-classifier"
RU_MODEL_NAME = "cointegrated/rubert-tiny-toxicity"

en_tokenizer = AutoTokenizer.from_pretrained(EN_MODEL_NAME)
en_model = AutoModelForSequenceClassification.from_pretrained(EN_MODEL_NAME)
en_model.eval()

ru_tokenizer = AutoTokenizer.from_pretrained(RU_MODEL_NAME)
ru_model = AutoModelForSequenceClassification.from_pretrained(RU_MODEL_NAME)
ru_model.eval()

# Метки моделей
EN_LABELS = ["toxic", "obscene", "insult", "threat", "identity_hate"]
RU_LABELS = ["non-toxic", "insult", "obscenity", "threat", "dangerous"]


class SafetyClassifierRepository(IMLServiceRepository):
    def __init__(self, mask: bool = True):
        self.mask = mask

    def mask_toxic_fragments(self, text: str, lang: str, threshold: float = 0.2) -> str:
        if lang not in {"en", "ru"}:
            return "Извини, не могу помочь тебе с этим вопросом"

        # Заменим исключения на маркеры
        protected_map = {}
        protected_counter = 0

        def protect(phrase):
            nonlocal protected_counter
            token = f"__PROTECTED_{protected_counter}__"
            protected_map[token] = phrase
            protected_counter += 1
            return token

        if lang == "en":
            exceptions = [
                "you",
                "you're",
                "you’re",
                "You",
                "You're",
                "YOU",
                "YOU’RE",
                "you are",
                "You are",
                "YOU ARE",
            ]
            for ex in sorted(exceptions, key=lambda x: -len(x)):  # длинные сначала
                text = text.replace(ex, protect(ex))

            words = text.split()
            masked_words = []
            for word in words:
                if word in protected_map:
                    masked_words.append(word)
                    continue
                inputs = en_tokenizer(
                    word, return_tensors="pt", truncation=True, padding=True
                )
                with torch.no_grad():
                    logits = en_model(**inputs).logits.squeeze()
                    probs = torch.sigmoid(logits).tolist()
                score = max(probs)
                masked_words.append("***" if score > threshold else word)

        elif lang == "ru":
            words = text.split()
            masked_words = []
            for word in words:
                inputs = ru_tokenizer(
                    word, return_tensors="pt", truncation=True, padding=True
                )
                with torch.no_grad():
                    logits = ru_model(**inputs).logits.squeeze()
                    probs = torch.sigmoid(logits).tolist()
                score = max(probs[1:])
                masked_words.append("*" * len(word) if score > threshold else word)

        # Восстанавливаем защищённые фразы
        final = " ".join(masked_words)
        for token, phrase in protected_map.items():
            final = final.replace(token, phrase)

        return final

    def process(self, message: BotMessage) -> ServiceCheckResult:
        txt = message.answer
        lang = detect_language(txt)

        if lang == "en":
            score_map = predict_toxicity_en(txt)
            tox_score = max(score_map.values())
        elif lang == "ru":
            score_map = predict_toxicity_ru(txt)
            tox_score = max([v for k, v in score_map.items() if k != "non-toxic"])
        else:
            tox_score = 0.0
            score_map = {"unknown": 0.0}
        safe = is_safe(tox_score)
        masked_message = txt
        if not (safe) and self.mask:
            masked_message = self.mask_toxic_fragments(txt, lang)
        elif not (safe) and not (self.mask):
            masked_message = "Извини, не могу помочь тебе с этим вопросом"
        return ServiceCheckResult(safe, tox_score, masked_message)


def is_safe(score) -> bool:
    if score >= 0.2:
        return False
    else:
        return True


def predict_toxicity_en(text: str) -> dict:
    inputs = en_tokenizer(
        text, return_tensors="pt", truncation=True, padding=True, max_length=512
    )
    with torch.no_grad():
        logits = en_model(**inputs).logits.squeeze()
        probs = torch.sigmoid(logits).tolist()
    return dict(zip(EN_LABELS, probs))


def predict_toxicity_ru(text: str) -> dict:
    inputs = ru_tokenizer(
        text, return_tensors="pt", truncation=True, padding=True, max_length=512
    )
    with torch.no_grad():
        logits = ru_model(**inputs).logits.squeeze()
        probs = torch.sigmoid(logits).tolist()
    return dict(zip(RU_LABELS, probs))
