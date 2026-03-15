from pydantic import BaseModel


class AnalyzeRequest(BaseModel):
    text: str
    url: str


class SourceResult(BaseModel):
    domain: str
    status: str
    label: int | None = None


class FactCheckResultItem(BaseModel):
    claim_text: str
    found: bool
    rating: str
    publisher: str
    url: str
    title: str
    textual_rating: str


class FactCheckResponse(BaseModel):
    claims_extracted: int
    claims_checked: int
    claims_with_results: int
    overall_score: float
    results: list[FactCheckResultItem]


class LinguisticResult(BaseModel):
    score: float
    confidence: float
    flags: list[str]
    explanation: str
    details: dict


class LimeWord(BaseModel):
    word: str
    weight: float
    direction: str


class AnalyzeResponse(BaseModel):
    prediction: str
    confidence: float
    fake_probability: float
    real_probability: float
    lime_explanation: list[LimeWord]
    source_verification: SourceResult
    fact_check: FactCheckResponse
    linguistic_analysis: LinguisticResult
