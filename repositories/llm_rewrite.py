from entities.data import LLMRewriteResult, BotMessage, LLMRequest
from use_cases.ports.ml_service import ILLMRewriteRepository
from gigachat import GigaChat


class GigachatRewriteRepository(ILLMRewriteRepository):
    def __init__(self):
        self.final_prompt = ""

    def process(self, message: BotMessage, request: LLMRequest) -> LLMRewriteResult:
        client = GigaChat(credentials=request.api_key, verify_ssl_certs=False)
        response = client.chat(request.prompt + message.answer)
        return LLMRewriteResult(
            answer=response.choices[0].message.content,
        )
