from entities.data import ServiceCheckResult, BotMessage, LLMRequest
from use_cases.ports.ml_service import ILLMRewriteRepository
from ollama import Client


class OllamaRewriteRepository(ILLMRewriteRepository):
    def __init__(self):
        self.final_prompt = ""

    def process(self, message: BotMessage, request: LLMRequest) -> ServiceCheckResult:
        self.final_prompt = request.prompt.replace("{answer}", message.answer)
        client = Client(host=request.ollama_host)
        response = client.chat(
            model=request.model,
            messages=[
                {
                    "role": "user",
                    "content": self.final_prompt,
                },
            ],
        )
        return ServiceCheckResult(
            safe=True,
            score=1,
            masked_answer=response["message"]["content"],
        )
