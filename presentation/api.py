# presentation/api.py
from fastapi import FastAPI, Depends, HTTPException
from entities.data import BotMessage, FinalCheckResult
from repositories.file_db import MongoResultRepository
from use_cases.check_message import CheckMessageUseCase
from use_cases.ports.event_bus import EventBus
from repositories.kafka_bus import KafkaEventBus
from config import settings

app = FastAPI(title="Output Safety API", debug=True)


async def get_event_bus() -> EventBus:
    return KafkaEventBus(brokers=settings.kafka_brokers)


async def get_enqueue_uc(bus: EventBus = Depends(get_event_bus)) -> CheckMessageUseCase:
    return CheckMessageUseCase(event_bus=bus)


@app.post("/check", status_code=202)
async def check_endpoint(
    payload: BotMessage,
    uc: CheckMessageUseCase = Depends(get_enqueue_uc),
):
    try:
        request_id = await uc.enqueue(payload)
        return {"request_id": request_id}
    except Exception as e:
        # send real error in dev; hide in prod
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/result/{request_id}", response_model=FinalCheckResult)
async def get_result(request_id: str):
    repo = MongoResultRepository(mongo_uri=settings.mongo_uri)
    doc = repo.collection.find_one({"request_id": request_id})
    if not doc or "result" not in doc:
        raise HTTPException(status_code=404, detail="Result not found")
    return doc["result"]
