import torch
import torch.nn.functional as F
from entities.data import ServiceCheckResult, BotMessage
from langdetect import detect, DetectorFactory
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from use_cases.ports.ml_service import IMLServiceRepository

"""
SAFETY CLASSIFIER MODULE

Purpose:
This module implements a safety classification service for Telegram bot messages to detect harmful, toxic, or disturbing content in user responses. It supports both English and Russian languages.

Models:
- English: "bert-base-uncased" — a general-purpose BERT model from HuggingFace used for binary classification (non-toxic vs. toxic).
- Russian: "DeepPavlov/rubert-base-cased" — a Russian-language BERT model from DeepPavlov. Supports multilabel classification for various toxicity types (e.g., obscene, insult, threat).

Pipeline Overview:
1. Language Detection — detects whether the message is in English or Russian using `langdetect`.
2. Tokenization — the input text is tokenized using the corresponding tokenizer for each model.
3. Classification:
   - For English: uses softmax on logits to obtain the probability of toxic content.
   - For Russian: uses sigmoid activation to evaluate multiple toxicity labels, then takes the highest probability among toxic labels.
4. Thresholding — if the toxicity score is above 0.3, the message is considered harmful.
5. Masking — if harmful, suspicious words are masked in the output (with "***") based on individual word-level toxicity estimation.

"""

DetectorFactory.seed = 42

# === Английская модель: HateBERT ===
en_model_name = "bert-base-uncased"
en_tokenizer = AutoTokenizer.from_pretrained(en_model_name)
en_model = AutoModelForSequenceClassification.from_pretrained(en_model_name)
en_model.eval()

# === Русская модель: ruBERT-Toxicity ===
ru_model_name = "DeepPavlov/rubert-base-cased"
ru_tokenizer = AutoTokenizer.from_pretrained(ru_model_name)
ru_model = AutoModelForSequenceClassification.from_pretrained(ru_model_name)
ru_model.eval()


class SafetyClassifierRepository(IMLServiceRepository):
    def process(self, message: BotMessage) -> ServiceCheckResult:
        txt = message.answer
        try:
            lang = detect(txt)
        except:
            lang = "unknown"
        if lang == "ru":
            score = predict_toxicity_ru(txt)
        elif lang == "en":
            score = predict_hate_speech_en(txt)

        else:
            score = 0.0
        if score >= 0.3:
            return ServiceCheckResult(False, score, mask_toxic_fragments(txt))
        else:
            return ServiceCheckResult(True, score, txt)


def predict_hate_speech_en(text: str) -> float:
    inputs = en_tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = en_model(**inputs)
        probs = F.softmax(outputs.logits, dim=1)
    return probs[0, 1].item()


def predict_toxicity_ru(text: str) -> float:
    inputs = ru_tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        logits = ru_model(**inputs).logits
        probs = torch.sigmoid(logits).squeeze().tolist()
    return max(probs[1:])  # пропускаем non-toxic


def mask_toxic_fragments(text: str, threshold: float = 0.5) -> str:
    try:
        lang = detect(text)
    except:
        lang = "unknown"

    words = text.split()
    masked = []

    for word in words:
        if lang == "ru":
            tokenizer = ru_tokenizer
            model = ru_model
            with torch.no_grad():
                inputs = tokenizer(word, return_tensors="pt", truncation=True, padding=True)
                logits = model(**inputs).logits
                probs = torch.sigmoid(logits).squeeze().tolist()
                toxicity_score = max(probs[1:])
        elif lang == "en":
            tokenizer = en_tokenizer
            model = en_model
            with torch.no_grad():
                inputs = tokenizer(word, return_tensors="pt", truncation=True, padding=True)
                logits = model(**inputs).logits
                probs = F.softmax(logits, dim=1)
                toxicity_score = probs[0, 1].item()
        else:
            masked.append(word)
            continue

        if toxicity_score > threshold:
            masked.append("***")
        else:
            masked.append(word)

    return ' '.join(masked)


# === Пример CLI-проверки ===
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
        serviceCheckResult = classifier.process(msg)
        print(f"📝 {msg.answer}")
        print(f"   ➤ Токсичность: {serviceCheckResult.score:.4f}")
        print(f"   ➤ Защита: {serviceCheckResult.masked_answer}\n")
