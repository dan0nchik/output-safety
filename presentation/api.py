from fastapi import FastAPI, HTTPException
from entities.data import BotMessage, CheckResult
from use_cases.check_message import CheckMessageUseCase
from repositories.llm_rewrite import LLMRewriteRepository
from repositories.pii_detector import PIIDetectorRepository
from repositories.ad_filter import AdFilterRepository
from repositories.off_topic_scorer import OffTopicRepository
from repositories.safety_classifier import SafetyClassifierRepository

app = FastAPI(title="Output Safety API", version="1.0")


@app.post("/check", response_model=CheckResult, summary="Check message safety")
async def check_endpoint(payload: BotMessage):
    try:
        pii = PIIDetectorRepository()
        safety = SafetyClassifierRepository()
        ad = AdFilterRepository()
        off_topic = OffTopicRepository()
        llm_rewrite = LLMRewriteRepository()
        check = CheckMessageUseCase(pii, safety, ad, off_topic, llm_rewrite)
        return check.execute(payload)
    except Exception:
        raise HTTPException(
            status_code=500, detail="Internal error while checking message"
        )
