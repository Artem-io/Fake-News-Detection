import re
import requests
from typing import List, Optional
from dataclasses import dataclass, field
from enum import Enum
from concurrent.futures import ThreadPoolExecutor, as_completed

from claim_extractor import ClaimExtractor


class FactCheckRating(Enum):
    TRUE = "TRUE"
    MOSTLY_TRUE = "MOSTLY_TRUE"
    MIXED = "MIXED"
    MOSTLY_FALSE = "MOSTLY_FALSE"
    FALSE = "FALSE"
    UNKNOWN = "UNKNOWN"


@dataclass
class FactCheckResult:
    claim_text: str
    found: bool
    rating: FactCheckRating = FactCheckRating.UNKNOWN
    publisher: str = ""
    url: str = ""
    title: str = ""
    textual_rating: str = ""


@dataclass
class ArticleFactCheckResult:
    claims_extracted: int
    claims_checked: int
    claims_with_results: int
    results: List[FactCheckResult] = field(default_factory=list)
    overall_score: float = 0.5

    def to_dict(self) -> dict:
        return {
            "claims_extracted": self.claims_extracted,
            "claims_checked": self.claims_checked,
            "claims_with_results": self.claims_with_results,
            "overall_score": self.overall_score,
            "results": [
                {
                    "claim_text": r.claim_text,
                    "found": r.found,
                    "rating": r.rating.value,
                    "publisher": r.publisher,
                    "url": r.url,
                    "title": r.title,
                    "textual_rating": r.textual_rating,
                }
                for r in self.results
            ],
        }


# Negation prefixes that invert a rating
_NEGATION_PREFIXES = re.compile(
    r'\b(not|no|lacks?|without|zero|isn\'?t|aren\'?t|wasn\'?t)\b',
    re.IGNORECASE
)

# Ordered from most specific to least specific — longer phrases must match first
# to prevent "mostly true" from matching "true" before "mostly true".
RATING_KEYWORDS = [
    (FactCheckRating.MOSTLY_TRUE, [
        "mostly true", "mostly correct", "mostly accurate",
    ]),
    (FactCheckRating.MOSTLY_FALSE, [
        "mostly false", "mostly wrong", "mostly incorrect",
    ]),
    (FactCheckRating.MIXED, [
        "mixed", "half true", "half false", "misleading",
        "out of context", "partly", "partially",
    ]),
    (FactCheckRating.FALSE, [
        "false", "fake", "incorrect", "wrong",
        "pants on fire", "pinocchio", "scam", "hoax", "debunked",
    ]),
    (FactCheckRating.TRUE, [
        "true", "correct", "accurate", "verified",
    ]),
]

_INVERSION_MAP = {
    FactCheckRating.TRUE: FactCheckRating.FALSE,
    FactCheckRating.MOSTLY_TRUE: FactCheckRating.MOSTLY_FALSE,
    FactCheckRating.FALSE: FactCheckRating.TRUE,
    FactCheckRating.MOSTLY_FALSE: FactCheckRating.MOSTLY_TRUE,
}

API_URL = "https://factchecktools.googleapis.com/v1alpha1/claims:search"

RATING_SCORES = {
    FactCheckRating.TRUE: 1.0,
    FactCheckRating.MOSTLY_TRUE: 0.75,
    FactCheckRating.MIXED: 0.5,
    FactCheckRating.MOSTLY_FALSE: 0.25,
    FactCheckRating.FALSE: 0.0,
}


def _normalize_rating(textual_rating: str) -> FactCheckRating:
    rating_lower = textual_rating.lower().strip()
    has_negation = bool(_NEGATION_PREFIXES.search(rating_lower))

    # Iterate in specificity order (RATING_KEYWORDS is a list of tuples)
    for fact_rating, keywords in RATING_KEYWORDS:
        for keyword in keywords:
            if keyword in rating_lower:
                if has_negation:
                    return _INVERSION_MAP.get(fact_rating, FactCheckRating.UNKNOWN)
                return fact_rating

    return FactCheckRating.UNKNOWN


def _search_google(api_key: str, query: str) -> List[FactCheckResult]:
    params = {"key": api_key, "query": query, "languageCode": "en"}
    try:
        response = requests.get(API_URL, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
    except requests.exceptions.RequestException:
        return []

    results = []
    for claim in data.get("claims", []):
        for review in claim.get("claimReview", []):
            textual_rating = review.get("textualRating", "")
            results.append(FactCheckResult(
                claim_text=claim.get("text", query),
                found=True,
                publisher=review.get("publisher", {}).get("name", ""),
                url=review.get("url", ""),
                title=review.get("title", ""),
                textual_rating=textual_rating,
                rating=_normalize_rating(textual_rating),
            ))
    return results


class FactChecker:
    def __init__(self, api_key: str, nlp=None):
        self.api_key = api_key
        self.extractor = ClaimExtractor(nlp=nlp)
        self._cache: dict[str, List[FactCheckResult]] = {}

        if not api_key:
            print("Warning: GOOGLE_FACTCHECK_API_KEY is empty. Fact-checking will return no results.")

    def _search_cached(self, query: str) -> List[FactCheckResult]:
        """Search with simple in-memory cache to avoid duplicate API calls."""
        cache_key = re.sub(r'\s+', ' ', query.lower().strip())
        if cache_key in self._cache:
            return self._cache[cache_key]
        results = _search_google(self.api_key, query)
        self._cache[cache_key] = results
        return results

    def _search_with_fallback(self, claim) -> List[FactCheckResult]:
        """
        Try multiple query strategies to maximize API hit rate:
        1. Simplified search_query (keywords only)
        2. Entity-focused query (just the named entities)
        3. Full sentence (original text, truncated)
        """
        # Strategy 1: simplified query (best match rate)
        results = self._search_cached(claim.search_query)
        if results:
            return results

        # Strategy 2: entity-only query (e.g. "Trump vaccine mandate")
        if claim.entity_names:
            entity_query = ' '.join(claim.entity_names[:4])
            if len(entity_query.split()) >= 2:
                results = self._search_cached(entity_query)
                if results:
                    return results

        # Strategy 3: first 8 words of original sentence
        short_original = ' '.join(claim.text.split()[:8])
        if short_original.lower().strip() != claim.search_query.lower().strip():
            results = self._search_cached(short_original)
            if results:
                return results

        return []

    def check_article(self, text: str, top_claims: int = 5) -> ArticleFactCheckResult:
        claims = self.extractor.extract(text, top_n=top_claims)

        if not claims:
            return ArticleFactCheckResult(
                claims_extracted=0, claims_checked=0, claims_with_results=0
            )

        if not self.api_key:
            return ArticleFactCheckResult(
                claims_extracted=len(claims),
                claims_checked=0,
                claims_with_results=0,
                results=[FactCheckResult(claim_text=c.text, found=False) for c in claims],
            )

        # Parallel API calls with multi-query fallback
        all_results = []
        claims_with_results = 0

        with ThreadPoolExecutor(max_workers=min(5, len(claims))) as executor:
            future_to_claim = {
                executor.submit(self._search_with_fallback, claim): claim
                for claim in claims
            }
            for future in as_completed(future_to_claim):
                claim = future_to_claim[future]
                try:
                    results = future.result()
                except Exception:
                    results = []

                if results:
                    claims_with_results += 1
                    all_results.extend(results)
                else:
                    all_results.append(FactCheckResult(claim_text=claim.text, found=False))

        # Weighted scoring: claims with more reviews get more weight
        weighted_sum = 0.0
        total_weight = 0.0
        for r in all_results:
            if r.found and r.rating in RATING_SCORES:
                weight = 1.0
                weighted_sum += RATING_SCORES[r.rating] * weight
                total_weight += weight

        overall_score = weighted_sum / total_weight if total_weight > 0 else 0.5

        return ArticleFactCheckResult(
            claims_extracted=len(claims),
            claims_checked=len(claims),
            claims_with_results=claims_with_results,
            results=all_results,
            overall_score=overall_score,
        )
