from pydantic import BaseModel


class AnalyzeRequest(BaseModel):
    text: str
    url: str


class ModuleDecision(BaseModel):
    module: str
    real_probability: float
    effective_weight: float
    has_data: bool
    reasoning: str


class LinguisticFlag(BaseModel):
    code: str
    description: str
    positive: bool | None


class LinguisticSignal(BaseModel):
    label: str
    value: str
    positive: bool | None


class LinguisticExplanation(BaseModel):
    verdict: str
    verdict_level: str   # RELIABLE / MIXED / CONCERNING / UNRELIABLE
    signals: list[LinguisticSignal]


class LinguisticAnalysis(BaseModel):
    score: float
    flags: list[LinguisticFlag]
    explanation: LinguisticExplanation



class MatchedSource(BaseModel):
    title: str
    url: str
    source: str
    published_at: str
    similarity: float


class CrossSourceSummary(BaseModel):
    verdict: str
    sources_found: int
    matched_sources: list[MatchedSource]


class AnalyzeResponse(BaseModel):
    prediction: str
    real_probability: float
    fake_probability: float
    decisions: list[ModuleDecision]
    source_status: str
    source_domain: str
    linguistic: LinguisticAnalysis
    cross_source: CrossSourceSummary
