import os
from contextlib import asynccontextmanager
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import spacy
from classifier import FakeNewsClassifier
from schemas import AnalyzeRequest, AnalyzeResponse
from source_verifier import SourceVerifier
from fact_checker import FactChecker
from linguistic_analyzer import LinguisticAnalyzer

load_dotenv()

# Shared spaCy model — loaded once, used by all components
shared_nlp = spacy.load("en_core_web_sm", disable=["lemmatizer"])
shared_nlp.max_length = 1_500_000

classifier: FakeNewsClassifier = None
verifier = SourceVerifier()
fact_checker = FactChecker(api_key=os.getenv("GOOGLE_FACTCHECK_API_KEY", ""), nlp=shared_nlp)
linguistic_analyzer = LinguisticAnalyzer(nlp=shared_nlp)


@asynccontextmanager
async def lifespan(app: FastAPI):
    global classifier
    classifier = FakeNewsClassifier('final_model')
    yield


app = FastAPI(title="Fake News Detector API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/analyze", response_model=AnalyzeResponse)
async def analyze(request: AnalyzeRequest):
    model_result = classifier.explain(request.text)
    source_result = verifier.check(request.url)
    fact_check_result = fact_checker.check_article(request.text)
    ling_result = linguistic_analyzer.analyze(request.text)
    return {
        **model_result,
        "source_verification": source_result,
        "fact_check": fact_check_result.to_dict(),
        "linguistic_analysis": {
            "score": ling_result.score,
            "confidence": ling_result.confidence,
            "flags": ling_result.flags,
            "explanation": ling_result.explanation,
            "details": ling_result.details,
        },
    }
