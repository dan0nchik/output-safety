# tests/infrastructure/adapters/test_off_topic_scorer.py

import pytest
from entities.data import BotMessage, ServiceCheckResult
from repositories.off_topic_scorer import OffTopicRepository


# Dummy tensor-like object to simulate .item() method
class DummyTensor:
    def __init__(self, value: float):
        self._value = value

    def item(self):
        return self._value


# Fixture to patch SentenceTransformer and util.cos_sim
@pytest.fixture(autouse=True)
def patch_sentence_transformers_and_util(monkeypatch):
    # Stub out the SentenceTransformer so it returns a dummy model
    class DummyModel:
        def __init__(self, model_name: str):
            # verify that the repository passes the correct model_name
            assert model_name == "test-model"

        def encode(self, text: str, convert_to_tensor: bool = True):
            # Return the text itself so we can inspect it if needed
            return f"vec-{text}"

    monkeypatch.setattr("repositories.off_topic_scorer.SentenceTransformer", DummyModel)

    # Stub out util.cos_sim to return a DummyTensor with controlled score
    def fake_cos_sim(a, b):
        # ensure that encode was called with correct inputs
        assert isinstance(a, str) and a.startswith("vec-")
        assert isinstance(b, str) and b.startswith("vec-")
        # for demonstration, return 0.8 similarity
        return DummyTensor(0.8)

    monkeypatch.setattr("repositories.off_topic_scorer.util.cos_sim", fake_cos_sim)


def test_process_marks_relevant_as_safe():
    repo = OffTopicRepository(model_name="test-model")
    msg = BotMessage(
        question="What is AI?", answer="AI stands for Artificial Intelligence."
    )

    result: ServiceCheckResult = repo.process(msg)

    # Since fake_cos_sim returns 0.8, and threshold is 0.5, safe should be True
    assert result.safe is True
    assert isinstance(result.score, float)
    assert result.score == pytest.approx(0.8)
    # The repository returns the original answer unchanged
    assert result.masked_answer == msg.answer


def test_process_marks_off_topic_as_unsafe(monkeypatch):
    # Override util.cos_sim to return a low score
    def low_cos_sim(a, b):
        return DummyTensor(0.3)

    monkeypatch.setattr("repositories.off_topic_scorer.util.cos_sim", low_cos_sim)

    repo = OffTopicRepository(model_name="test-model")
    msg = BotMessage(question="Explain relativity.", answer="Bananas are yellow.")

    result: ServiceCheckResult = repo.process(msg)

    # 0.3 < 0.5 threshold, so this is off-topic
    assert result.safe is False
    assert result.score == pytest.approx(0.3)
    assert result.masked_answer == msg.answer
