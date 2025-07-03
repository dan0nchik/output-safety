from entities.data import BotMessage, CheckResult
from repositories.ad_filter import AdFilterRepository
from repositories.llm_rewrite import LLMRewriteRepository
from repositories.off_topic_scorer import OffTopicRepository
from repositories.pii_detector import PIIDetectorRepository
from repositories.safety_classifier import SafetyClassifierRepository


class CheckMessageUseCase:
    def __init__(
        self,
        pii: PIIDetectorRepository,
        safety: SafetyClassifierRepository,
        ad: AdFilterRepository,
        off_topic: OffTopicRepository,
        rewrite: LLMRewriteRepository,
    ):
        self.llm_rewrite = rewrite

    def execute(self, message: BotMessage) -> CheckResult:
        r1 = self.llm_rewrite.process("Hello")
        return CheckResult(True, [], 0, "", message.answer)
