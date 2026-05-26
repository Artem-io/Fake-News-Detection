import json
import os
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import spacy
from classifier import FakeNewsClassifier
from schemas import AnalyzeRequest, AnalyzeResponse
from source_verifier import SourceVerifier
from linguistic_analyzer import LinguisticAnalyzer
from source_comparator import SourceComparator
from aggregator import aggregate

LOG_PATH = Path(__file__).parent / "analysis_log.jsonl"

load_dotenv()

shared_nlp = spacy.load("en_core_web_sm", disable=["lemmatizer"])
shared_nlp.max_length = 1_500_000

classifier: FakeNewsClassifier = None
verifier = SourceVerifier()
linguistic_analyzer = LinguisticAnalyzer(nlp=shared_nlp)
source_comparator = SourceComparator(nlp=shared_nlp)


@asynccontextmanager
async def lifespan(app: FastAPI):
    global classifier
    classifier = FakeNewsClassifier(os.getenv("MODEL_ID", "Artemi0/fake-news-distilbert"))
    yield


app = FastAPI(title="Fake News Detector API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def _build_response(model_result, source_result, ling_result, cross_source_result, aggregated):
    return {
        "prediction": aggregated.prediction,
        "real_probability": aggregated.real_probability,
        "fake_probability": aggregated.fake_probability,
        "decisions": aggregated.decisions,
        "source_status": source_result.get("status", "NOT_FOUND"),
        "source_domain": source_result.get("domain", ""),
        "linguistic": {
            "score": ling_result.score,
            "flags": ling_result.flags,
            "explanation": ling_result.explanation,
            "headline_consistency": ling_result.headline_consistency,
        },
        "cross_source": {
            "verdict": cross_source_result.verdict,
            "sources_found": cross_source_result.sources_found,
            "matched_sources": [
                {
                    "title": s.title,
                    "url": s.url,
                    "source": s.source,
                    "published_at": s.published_at,
                    "similarity": s.similarity,
                }
                for s in cross_source_result.matched_sources
            ],
        },
    }


async def _run_all_modules(request: AnalyzeRequest):
    full_text = f"{request.title} {request.text}" if request.title else request.text
    model_result       = classifier.predict(full_text)
    source_result      = verifier.check(request.url)
    ling_result        = linguistic_analyzer.analyze(request.text, title=request.title)
    cross_source_result = source_comparator.compare(request.text, title=request.title, url=request.url)
    return model_result, source_result, ling_result, cross_source_result


def _append_log(request: AnalyzeRequest, model_result, source_result, ling_result, aggregated) -> None:
    record = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "url": request.url,
        "title": request.title,
        # Combined
        "combined": {
            "prediction": aggregated.prediction,
            "real_probability": aggregated.real_probability,
            "fake_probability": aggregated.fake_probability,
        },
        # DistilBERT
        "distilbert": {
            "prediction": model_result["prediction"],
            "real_probability": round(model_result["real_probability"], 4),
            "fake_probability": round(model_result["fake_probability"], 4),
            "confidence": round(model_result["confidence"], 4),
        },
        # Source
        "source": {
            "domain": source_result.get("domain", ""),
            "status": source_result.get("status", "NOT_FOUND"),
        },
        # Linguistic
        "linguistic": {
            "score": ling_result.score,
            "verdict_level": ling_result.explanation.get("verdict_level", ""),
            "headline_consistency": ling_result.headline_consistency,
            "flags": [f["code"] for f in ling_result.flags],
        },
    }
    with LOG_PATH.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, ensure_ascii=False) + "\n")


@app.post("/analyze", response_model=AnalyzeResponse)
async def analyze(request: AnalyzeRequest):
    model_result, source_result, ling_result, cross_source_result = await _run_all_modules(request)
    aggregated = aggregate(
        model_result=model_result,
        source_result=source_result,
        ling_result=ling_result,
    )
    _append_log(request, model_result, source_result, ling_result, aggregated)
    return _build_response(model_result, source_result, ling_result, cross_source_result, aggregated)
