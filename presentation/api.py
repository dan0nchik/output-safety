from fastapi import FastAPI, HTTPException, Depends
from entities.data import BotMessage, LLMRequest, FinalCheckResult
from use_cases.check_message import CheckMessageUseCase
from repositories.llm_rewrite import OllamaRewriteRepository
from repositories.pii_detector import PIIDetectorRepository
from repositories.ad_filter import AdFilterRepository
from repositories.off_topic_scorer import OffTopicRepository
from repositories.safety_classifier import SafetyClassifierRepository
from config import settings

app = FastAPI(title="Output Safety API", version="1.0")


def get_check_use_case() -> CheckMessageUseCase:
    pii = PIIDetectorRepository()
    safety = SafetyClassifierRepository()
    ad = AdFilterRepository()
    off_topic = OffTopicRepository()
    llm_rewrite = OllamaRewriteRepository()
    llm_request = LLMRequest(
        prompt=settings.ollama_prompt,
        model=settings.ollama_model_name,
        ollama_host=settings.ollama_base_url,
        api_key=None,
    )
    return CheckMessageUseCase(pii, safety, ad, off_topic, llm_rewrite, llm_request)


@app.post("/check", response_model=FinalCheckResult, summary="Check message safety")
async def check_endpoint(
    payload: BotMessage, use_case: CheckMessageUseCase = Depends(get_check_use_case)
):
    try:
        return use_case.execute(payload)
    except Exception:
        raise HTTPException(
            status_code=500, detail="Internal error while checking message"
        )
