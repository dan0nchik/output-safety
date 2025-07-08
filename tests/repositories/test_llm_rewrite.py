# tests/infrastructure/adapters/test_ollama_rewrite.py
import pytest

from entities.data import BotMessage, LLMRequest, ServiceCheckResult
from repositories.llm_rewrite import OllamaRewriteRepository


class DummyClient:
    def __init__(self, host: str):
        # capture the host passed in
        self.host = host

    def chat(self, model: str, messages: list[dict]):
        # Assert that the model and message content are forwarded correctly
        assert model == "test-model"
        # messages is a list of dicts with "role" and "content"
        assert messages == [{"role": "user", "content": "PROMPT: original answer"}]
        # Return a fake API response
        return {"message": {"content": "rewritten answer"}}


@pytest.fixture(autouse=True)
def patch_ollama_client(monkeypatch):
    """
    Monkeypatch ollama.Client to use our DummyClient instead of
    making real network calls.
    """
    monkeypatch.setattr("repositories.llm_rewrite.Client", DummyClient)


def test_process_rewrites_and_returns_correct_result():
    # Arrange: create repository, message and request
    repo = OllamaRewriteRepository()
    msg = BotMessage(
        question="Q?",
        answer="original answer",
    )
    req = LLMRequest(
        prompt="PROMPT: {answer}",
        model="test-model",
        ollama_host="https://fake-ollama",
        api_key="fake-key",
    )

    # Act
    result: ServiceCheckResult = repo.process(msg, req)

    # Assert: prompt was rendered correctly
    assert repo.final_prompt == "PROMPT: original answer"

    # Assert: correct ServiceCheckResult fields
    assert isinstance(result, ServiceCheckResult)
    assert result.safe is True
    assert result.score == 1
    assert result.masked_answer == "rewritten answer"
