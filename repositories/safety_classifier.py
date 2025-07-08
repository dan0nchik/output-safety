import torch
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

detector = LanguageDetectorBuilder.from_languages(Language.ENGLISH, Language.RUSSIAN).build()

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
        return ServiceCheckResult(is_toxic(tox_score),tox_score,txt)

def is_toxic(score) -> bool:
    if score >= 0.1:
        return False
    else:
        return True
def predict_toxicity_en(text: str) -> dict:
    inputs = en_tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        logits = en_model(**inputs).logits.squeeze()
        probs = torch.sigmoid(logits).tolist()
    return dict(zip(EN_LABELS, probs))

def predict_toxicity_ru(text: str) -> dict:
    inputs = ru_tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        logits = ru_model(**inputs).logits.squeeze()
        probs = torch.sigmoid(logits).tolist()
    return dict(zip(RU_LABELS, probs))

def mask_toxic_fragments(text: str, lang: str, threshold: float = 0.5) -> str:
    words = text.split()
    masked = []

    for word in words:
        if lang == "en":
            inputs = en_tokenizer(word, return_tensors="pt", truncation=True, padding=True)
            with torch.no_grad():
                logits = en_model(**inputs).logits.squeeze()
                probs = torch.sigmoid(logits).tolist()
            score = max(probs)
        elif lang == "ru":
            inputs = ru_tokenizer(word, return_tensors="pt", truncation=True, padding=True)
            with torch.no_grad():
                logits = ru_model(**inputs).logits.squeeze()
                probs = torch.sigmoid(logits).tolist()
            score = max(probs[1:])  # пропускаем "non-toxic"
        else:
            masked.append(word)
            continue

        masked.append("***" if score > threshold else word)

    return " ".join(masked)

if __name__ == "__main__":
    test_messages = [
        BotMessage(question="Тестовое сообщение",
                   answer="I can hear voices in the walls at night.There was blood everywhere. I couldn’t stop shaking.He said he’d come for me in my sleep.They found a body in the woods, missing its face.I had a dream where everyone I loved was dead.The smell of burnt flesh still lingers in my mind.She laughed while setting the house on fire.They tortured him for hours. No one stopped them."),
        BotMessage(question="Тестовое сообщение",
                   answer="Всю стену залило кровью, я не мог дышать от ужаса.Он сказал, что убьёт меня, если я расскажу.Я слышу шаги по ночам, но дома я один.Тело нашли в подвале. Без глаз.Мне снятся сны, где я умираю каждый раз по-новому.У неё был нож, и она улыбалась.Он кричал, но никто не пришёл.Мама плакала, когда увидела, что я нарисовал."),
        BotMessage(question="Тестовое сообщение", answer="Ты молодец"),
        BotMessage(question="Тестовое сообщение", answer="Go away"),
        BotMessage(question="Тестовое сообщение", answer="Иди к чёрту"),
    ]

    classifier = SafetyClassifierRepository()
    for msg in test_messages:
        print(f"📝 {msg.answer}")
        print( classifier.process(msg).score)
