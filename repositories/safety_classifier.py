import logging
import torch
from entities.data import ServiceCheckResult, BotMessage
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from use_cases.ports.ml_service import IMLServiceRepository
from lingua import Language, LanguageDetectorBuilder

# Configure logging
torch.set_num_threads(1)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Language detection setup
detector = LanguageDetectorBuilder.from_languages(
    Language.ENGLISH, Language.RUSSIAN
).build()


def detect_language(text: str) -> str:
    """
    Detects text language with confidence threshold.
    Returns 'en', 'ru', or 'unknown'.
    """
    if not text or not text.strip():
        return "unknown"
    # compute confidence values
    confidences = detector.compute_language_confidence_values(text)
    # map to (lang, value)
    best = max(confidences, key=lambda c: c.value)
    logger.debug(
        "Language confidences: %s", {str(c.language): c.value for c in confidences}
    )
    # require at least 50% confidence
    if best.value < 0.5:
        logger.warning("Low language detection confidence: %s", best)
        return "unknown"
    return "en" if best.language == Language.ENGLISH else "ru"


# Model names
EN_MODEL_NAME = "ujjawalsah/bert-toxicity-classifier"
RU_MODEL_NAME = "cointegrated/rubert-tiny-toxicity"

# Load English model and tokenizer
en_tokenizer = AutoTokenizer.from_pretrained(EN_MODEL_NAME)
en_model = AutoModelForSequenceClassification.from_pretrained(EN_MODEL_NAME)
en_model.eval()

# Load Russian model and tokenizer
ru_tokenizer = AutoTokenizer.from_pretrained(RU_MODEL_NAME)
ru_model = AutoModelForSequenceClassification.from_pretrained(RU_MODEL_NAME)
ru_model.eval()

# Labels
en_labels = ["toxic", "obscene", "insult", "threat", "identity_hate"]
ru_labels = ["non-toxic", "insult", "obscenity", "threat", "dangerous"]


def safe_sigmoid(logits: torch.Tensor) -> list:
    """
    Apply sigmoid and return a list of probabilities for each label.
    Handles both batched and unbatched logits safely.
    """
    if logits.ndim == 2 and logits.size(0) == 1:
        logits = logits[0]
    probs = torch.sigmoid(logits)
    if probs.ndim == 0:
        return [probs.item()]
    return probs.tolist()


def predict_toxicity_en(text: str) -> dict:
    try:
        inputs = en_tokenizer(
            text, return_tensors="pt", truncation=True, padding=True, max_length=512
        )
        with torch.no_grad():
            logits = en_model(**inputs).logits
        probs = safe_sigmoid(logits)
        if len(probs) != len(en_labels):
            logger.warning(
                "EN model returned %d probabilities, expected %d",
                len(probs),
                len(en_labels),
            )
            probs = (probs + [0.0] * len(en_labels))[: len(en_labels)]
        return dict(zip(en_labels, probs))
    except Exception as e:
        logger.error("Error in English toxicity prediction: %s", e)
        return {label: 0.0 for label in en_labels}


def predict_toxicity_ru(text: str) -> dict:
    try:
        # Token-level check for gibberish/out-of-vocab
        tokens = ru_tokenizer.tokenize(text)
        logger.debug("RU tokens for '%s': %s", text, tokens)
        # If all tokens are unknown or too few tokens, skip model
        if (
            not tokens
            or all(t == ru_tokenizer.unk_token for t in tokens)
            or len(tokens) < 2
        ):
            logger.warning("Skipping gibberish or OOV input: %s", tokens)
            return {label: 0.0 for label in ru_labels}

        inputs = ru_tokenizer(
            text, return_tensors="pt", truncation=True, padding=True, max_length=512
        )
        with torch.no_grad():
            logits = ru_model(**inputs).logits
        probs = safe_sigmoid(logits)
        if len(probs) != len(ru_labels):
            logger.warning(
                "RU model returned %d probabilities, expected %d",
                len(probs),
                len(ru_labels),
            )
            probs = (probs + [0.0] * len(ru_labels))[: len(ru_labels)]
        return dict(zip(ru_labels, probs))
    except Exception as e:
        logger.error("Error in Russian toxicity prediction: %s", e)
        return {label: 0.0 for label in ru_labels}


def is_toxic(score: float, threshold: float = 0.1) -> bool:
    # True => safe (below threshold), False => toxic
    return score < threshold


class SafetyClassifierRepository(IMLServiceRepository):
    def process(self, message: BotMessage) -> ServiceCheckResult:
        txt = message.answer or ""
        lang = detect_language(txt)
        logger.info(
            "Processing text of length %d, detected language: %s", len(txt), lang
        )

        if lang == "en":
            scores = predict_toxicity_en(txt)
            tox_score = max(scores.values())
        elif lang == "ru":
            scores = predict_toxicity_ru(txt)
            tox_score = max(
                (v for k, v in scores.items() if k != "non-toxic"), default=0.0
            )
        else:
            scores = {"unknown": 0.0}
            tox_score = 0.0

        safe_flag = is_toxic(tox_score)
        logger.info(
            "Result safe=%s, score=%.4f, scores=%s", safe_flag, tox_score, scores
        )
        return ServiceCheckResult(safe=safe_flag, score=tox_score, masked_answer=txt)


def mask_toxic_fragments(text: str, lang: str, threshold: float = 0.5) -> str:
    words = text.split()
    masked = []
    for word in words:
        try:
            if lang == "en":
                probs = safe_sigmoid(
                    en_model(**en_tokenizer(word, return_tensors="pt")).logits
                )
                score = max(probs)
            elif lang == "ru":
                probs = safe_sigmoid(
                    ru_model(**ru_tokenizer(word, return_tensors="pt")).logits
                )
                score = max(probs[1:])  # skip "non-toxic"
            else:
                masked.append(word)
                continue
            masked.append("***" if score > threshold else word)
        except Exception:
            masked.append(word)
    return " ".join(masked)


# Self-test when run as a script
if __name__ == "__main__":
    test_msgs = [
        BotMessage(question="Q", answer="Ты молодец"),
        BotMessage(question="Q", answer="Иди к чёрту"),
        BotMessage(question="Q", answer="ПАСАСИ"),
        BotMessage(question="Q", answer="привет"),
    ]
    repo = SafetyClassifierRepository()
    for m in test_msgs:
        result = repo.process(m)
        print(f"Answer: {m.answer}, Score: {result.score}, Safe: {result.safe}")
