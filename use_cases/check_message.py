from entities.data import BotMessage, FinalCheckResult, LLMRequest, ServiceCheckResult
from use_cases.ports.ml_service import IMLServiceRepository, ILLMRewriteRepository


class CheckMessageUseCase:
    def __init__(
        self,
        pii: IMLServiceRepository,
        safety: IMLServiceRepository,
        ad: IMLServiceRepository,
        off_topic: IMLServiceRepository,
        rewrite: ILLMRewriteRepository,
        llm_request: LLMRequest,
    ):
        self.llm_rewrite = rewrite
        self.pii = pii
        self.safety = safety
        self.ad = ad
        self.off_topic = off_topic
        self.llm_request = llm_request

    def execute(self, message: BotMessage) -> FinalCheckResult:
        # off_topic_check: ServiceCheckResult = self.off_topic.process(message)
        # ad_filter_check: ServiceCheckResult = self.ad.process(message)
        # safety_check: ServiceCheckResult = self.safety.process(message)
        # pii_check: ServiceCheckResult = self.pii.process(message)
        # print("offtopic", off_topic_check)
        # print("ad", ad_filter_check)
        # print("safety", safety_check)
        # print("pi", pii_check)
        return FinalCheckResult(safe=True, violations=[], score=0, masked_answer="")
