from entities.data import ServiceCheckResult, BotMessage
from use_cases.ports.ml_service import IMLServiceRepository
import re
from typing import List
from deeppavlov import build_model, configs
import torchcrf


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

    # --- DeepPavlov NERUS initialization ---
    _ner_model = build_model(
        configs.ner.ner_ontonotes_bert_mult, download=True, install=True
    )

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
        result = self._ner_model([text])
        tokens, tags = result[0][0], result[1][0]
        found = []
        current = []
        start = None
        idx_in_text = 0
        i_person_counter = 0
        otchestvo_span = None
        for idx, (token, tag) in enumerate(zip(tokens, tags)):
            is_b = tag in ("B-PER", "B-PERSON")
            is_i = tag in ("I-PER", "I-PERSON")
            if is_b:
                # Завершаем предыдущую последовательность
                if otchestvo_span:
                    found.append(
                        {
                            "type": "FIO_OTCHESTVO",
                            "match": otchestvo_span["match"],
                            "span": otchestvo_span["span"],
                        }
                    )
                    otchestvo_span = None
                current = [token]
                i_person_counter = 0
                # Позиция фамилии
                start = text.find(token, idx_in_text)
                idx_in_text = start + len(token)
            elif is_i and current:
                i_person_counter += 1
                idx_in_text = text.find(token, idx_in_text)
                if idx_in_text != -1:
                    idx_in_text += len(token)
                current.append(token)
                # Если это второй I-PERSON/I-PER — считаем это отчеством
                if i_person_counter == 2:
                    otchestvo_start = text.find(token, idx_in_text - len(token))
                    otchestvo_end = otchestvo_start + len(token)
                    otchestvo_span = {
                        "type": "FIO_OTCHESTVO",
                        "match": token,
                        "span": (otchestvo_start, otchestvo_end),
                    }
            else:
                if otchestvo_span:
                    found.append(
                        {
                            "type": "FIO_OTCHESTVO",
                            "match": otchestvo_span["match"],
                            "span": otchestvo_span["span"],
                        }
                    )
                    otchestvo_span = None
                current = []
                i_person_counter = 0
        if otchestvo_span:
            found.append(
                {
                    "type": "FIO_OTCHESTVO",
                    "match": otchestvo_span["match"],
                    "span": otchestvo_span["span"],
                }
            )
        return found

    def _mask_text(self, text: str, pii_matches: List[dict]) -> str:
        masked = list(text)
        for match in pii_matches:
            start, end = match["span"]
            if (
                match["type"] == "PASSPORT"
                or match["type"] == "PASSPORT_SERIES"
                or match["type"] == "PASSPORT_NUMBER"
            ):
                for i in range(start, end):
                    masked[i] = "X"
            elif match["type"] == "PHONE":
                # Маскируем только сам номер, не слово 'номер'
                # Проверяем, что маскируем только последовательность цифр (и +, если есть)
                phone_text = match["match"]
                phone_start = text.find(phone_text, start)
                if phone_start != -1:
                    for i in range(phone_start, phone_start + len(phone_text)):
                        masked[i] = "X"
            elif match["type"] == "EMAIL":
                for i in range(start, end):
                    masked[i] = "x"
            elif match["type"] == "FIO_OTCHESTVO":
                for i in range(start, end):
                    masked[i] = "*"
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

        texts = [("question", message.question), ("answer", message.answer)]
        all_matches = []
        violations = []
        masked_answer = message.answer
        max_ratio = 0.0
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
                            level=ViolationLevel.HIGH
                            if m["type"]
                            in [
                                "PASSPORT",
                                "PASSPORT_SERIES",
                                "PASSPORT_NUMBER",
                                "PHONE",
                            ]
                            else ViolationLevel.MEDIUM,
                        )
                    )
            if field == "answer" and matches:
                masked_answer = self._mask_text(text, matches)
        safe = not bool(all_matches)
        score = int(max_ratio * 100)
        actions = "mask" if not safe else "none"
        return ServiceCheckResult(safe=safe, score=score, masked_answer=masked_answer)


if __name__ == "__main__":
    from entities.data import BotMessage

    # Интерактивный режим
    question = input("Введите вопрос: ")
    answer = input("Введите ответ: ")
    message = BotMessage(question=question, answer=answer)
    detector = PIIDetectorRepository()
    result = detector.process(message)
    print("Safe:", result.safe)
    print("Score:", result.score)
    print("Masked answer:", result.masked_answer)
