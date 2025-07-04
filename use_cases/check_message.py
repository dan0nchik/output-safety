from entities.data import BotMessage, FinalCheckResult, LLMRequest
from ports.ml_service import IMLServiceRepository, ILLMRewriteRepository


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
        # TODO проверка decision engine
        llm_check = self.llm_rewrite.process(message, self.llm_request)
        return FinalCheckResult()
