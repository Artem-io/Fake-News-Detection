from contextlib import asynccontextmanager
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
    model_result = classifier.predict(request.text)
    source_result = verifier.check(request.url)
    ling_result = linguistic_analyzer.analyze(request.text)
    cross_source_result = source_comparator.compare(request.text)

    aggregated = aggregate(
        model_result=model_result,
        source_result=source_result,
        ling_result=ling_result,
    )

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
