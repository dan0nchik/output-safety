from entities.data import CheckResult, BotMessage
from use_cases.ports.ml_service import IMLServiceRepository

import re
import string
from typing import List, Dict

# Для русского: pip install pymorphy2 ru-core-news-sm spacy
import spacy
from pymorphy2 import MorphAnalyzer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

"""
SERVICE FOR DETECTING AD & COMPETITOR INFORMATION

PIPELINE:
1. Detection of the message's language (Russian / English)
2. Deletion of links, HTML-inserts and some special symbols
3. Tokenization of the text
4. Running rule engine (cheching 3 categories: promo / competiton / common text)
5. Running TF-IDF + linear classification (cheching 3 categories)
"""

# example rules for 'rule engine'
RULES = {
    'promo': [r'\bскидк[а-я]*\b', r'\bакция\b', r'\bподарок\b'],
    'competitor': [r'\bбренд1\b', r'\bконкурент2\b', r'\bcompetitor-site\.com\b']
}

class TextPreprocessor:
    def __init__(self, language: str = 'ru'):
        if language == 'ru':
            self.nlp = spacy.load('ru_core_news_sm')
            self.morph = MorphAnalyzer()
            self.stopwords = spacy.lang.ru.stop_words.STOP_WORDS
        else:
            self.nlp = spacy.load('en_core_web_sm')
            self.morph = None
            self.stopwords = spacy.lang.en.stop_words.STOP_WORDS

    def clean_text(self, text: str) -> str:
        text = re.sub(r'<[^>]+>', ' ', text)
        text = re.sub(r'http\S+|www\.\S+', ' ', text)
        text = text.translate(str.maketrans('', '', string.punctuation))
        text = text.lower()
        return text

    def tokenize(self, text: str) -> List[str]:
        doc = self.nlp(text)
        tokens = []
        for token in doc:
            if token.is_space or token.is_punct or token.is_digit:
                continue
            lemma = token.lemma_.strip()
            if not lemma or lemma in self.stopwords:
                continue
            if self.morph:
                lemma = self.morph.parse(lemma)[0].normal_form
            tokens.append(lemma)
        return tokens

    def preprocess(self, text: str) -> str:
        cleaned = self.clean_text(text)
        tokens = self.tokenize(cleaned)
        return ' '.join(tokens)
    
class RuleEngine:
    def __init__(self, rules: Dict[str, List[str]]):
        self.compiled = {
            label: [re.compile(pat, re.IGNORECASE) for pat in patterns]
            for label, patterns in rules.items()
        }

    def apply(self, text: str) -> List[str]:
        matches = []
        for label, patterns in self.compiled.items():
            for pat in patterns:
                if pat.search(text):
                    matches.append(label)
                    break
        return matches
    
def build_ml_pipeline() -> Pipeline:
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer()),
        ('clf', LogisticRegression(solver='liblinear'))
    ])
    return pipeline

class AdFilterRepository(IMLServiceRepository):
    def process(self, message: BotMessage) -> CheckResult:
        preproc = TextPreprocessor(language='ru')
        rules = RuleEngine(RULES)
        ml_pipeline = build_ml_pipeline()

        cleaned_texts = [preproc.preprocess(t) for t in message]
        labels = ['promo', 'competitor', 'normal']
        # TODO: THE SEPERATE SERVICE FOR MODEL'S FITTING

        # TODO: APPLIEMENT OF THE AD & COMPETITOR SERVICE
