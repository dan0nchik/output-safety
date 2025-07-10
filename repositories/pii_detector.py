from entities.data import ServiceCheckResult, BotMessage
from use_cases.ports.ml_service import IMLServiceRepository
import re
from typing import List
from transformers import pipeline


class PIIDetectorRepository(IMLServiceRepository):
    # Ключевые слова для телефона и паспорта
    PHONE_WORDS = [
        "номер телефона",
        "номера телефона",
        "номеру телефона",
        "номером телефона",
        "номере телефона",
    ]
    PASSPORT_WORDS = [
        "паспорт",
        "паспортные данные",
        "паспортных данных",
        "паспортными данными",
        "серия",
        "номер",
    ]
    # Регулярки
    PHONE_REGEX = re.compile(r"(\+7|8|7)\d{10}")
    EMAIL_REGEX = re.compile(r"[\w\.-]+@[\w\.-]+\.[a-zA-Z]{2,}")
    PASSPORT_REGEX = re.compile(r"(\b\d{4}\s?\d{6}\b|\b\d{10}\b)")
    PASSPORT_SERIES_REGEX = re.compile(r"серия\s*\d{4}", re.IGNORECASE)
    PASSPORT_NUMBER_REGEX = re.compile(r"номер\s*\d{6}", re.IGNORECASE)
    print("PIIDetector initialized with regex patterns")
    # --- HuggingFace NER pipeline initialization ---
    _ner_pipe = pipeline("ner", model="Gherman/bert-base-NER-Russian")
    print("NER pipeline initialized")

    def _find_phone(self, text: str) -> List[dict]:
        found = []
        for match in self.PHONE_REGEX.finditer(text):
            found.append(
                {"type": "PHONE", "match": match.group(), "span": match.span()}
            )
        return found

    def _find_email(self, text: str) -> List[dict]:
        return [
            {"type": "EMAIL", "match": m.group(), "span": m.span()}
            for m in self.EMAIL_REGEX.finditer(text)
        ]

    def _find_passport(self, text: str) -> List[dict]:
        found = []
        if any(word in text.lower() for word in self.PASSPORT_WORDS):
            for match in self.PASSPORT_REGEX.finditer(text):
                found.append(
                    {"type": "PASSPORT", "match": match.group(), "span": match.span()}
                )
            for match in self.PASSPORT_SERIES_REGEX.finditer(text):
                found.append(
                    {
                        "type": "PASSPORT_SERIES",
                        "match": match.group(),
                        "span": match.span(),
                    }
                )
            for match in self.PASSPORT_NUMBER_REGEX.finditer(text):
                found.append(
                    {
                        "type": "PASSPORT_NUMBER",
                        "match": match.group(),
                        "span": match.span(),
                    }
                )
        return found

    def _find_fio(self, text: str) -> List[dict]:
        results = self._ner_pipe(text)
        print("NER results:", results)  # For debugging

        found = []
        current = None

        for res in results:
            tag = res["entity"].replace("U-", "").replace("B-", "").replace("I-", "")
            word = res["word"]
            start = res["start"]
            end = res["end"]

            if word.startswith("##") and current:
                current["match"] += word[2:]
                current["span"] = (current["span"][0], end)
            else:
                # Push previous match if any
                if current:
                    found.append(current)
                if tag in ["FIRST_NAME", "LAST_NAME", "MIDDLE_NAME"]:
                    current = {"type": tag, "match": word, "span": (start, end)}
                else:
                    current = None

        # Final entity
        if current:
            found.append(current)

        return found

    def _mask_text(self, text: str, pii_matches: List[dict]) -> str:
        masked = list(text)
        for match in pii_matches:
            start, end = match["span"]
            mtype = match["type"]

            if mtype in ["PASSPORT", "PASSPORT_SERIES", "PASSPORT_NUMBER", "PHONE"]:
                for i in range(start, end):
                    masked[i] = "*"
            elif mtype == "EMAIL":
                for i in range(start, end):
                    masked[i] = "*"
            elif mtype in ["LAST_NAME", "MIDDLE_NAME"]:  # Mask only these
                for i in range(start, end):
                    masked[i] = "*"
            # FIRST_NAME is deliberately skipped
        return "".join(masked)

    def _pii_word_ratio(self, text: str, pii_matches: List[dict]) -> float:
        words = re.findall(r"\w+", text)
        if not words:
            return 0.0
        pii_words = set()
        for match in pii_matches:
            for w in re.findall(r"\w+", match["match"]):
                pii_words.add(w)
        count = sum(1 for w in words if w in pii_words)
        return count / len(words) if words else 0.0

    def process(self, message: BotMessage) -> ServiceCheckResult:
        from entities.data import Violation, ViolationLevel

        print("Processing message:", message)
        texts = [("question", message.question), ("answer", message.answer)]
        all_matches = []
        violations = []
        masked_answer = message.answer
        max_ratio = 0.0
        censored_types = set()
        for field, text in texts:
            matches = []
            matches += self._find_phone(text)
            matches += self._find_email(text)
            matches += self._find_passport(text)
            matches += self._find_fio(text)
            if matches:
                all_matches.extend(matches)
                ratio = self._pii_word_ratio(text, matches)
                max_ratio = max(max_ratio, ratio)
                for m in matches:
                    violations.append(
                        Violation(
                            violation_type=m["type"],
                            level=(
                                ViolationLevel.HIGH
                                if m["type"]
                                in [
                                    "PASSPORT",
                                    "PASSPORT_SERIES",
                                    "PASSPORT_NUMBER",
                                    "PHONE",
                                ]
                                else ViolationLevel.MEDIUM
                            ),
                        )
                    )
                    censored_types.add(m["type"])
                masked_answer = self._mask_text(text, matches)
        safe = not bool(all_matches)
        score = max_ratio
        actions = "mask" if not safe else "none"
        return ServiceCheckResult(safe=safe, score=score, masked_answer=masked_answer)
