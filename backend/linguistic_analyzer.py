import re
import warnings
from dataclasses import dataclass, field
from typing import List, Dict, Optional
from collections import Counter
import spacy
import textstat
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

"""
Features analyzed:
1. Clickbait patterns
2. Emotional/sensational language
3. Exclamation marks overuse
4. Vague attribution ("sources say")
5. Urgency/fear words
6. Superlatives and absolutes
7. Credibility indicators (quotes, named sources)
8. Conspiracy/propaganda language
9. Hedging words
10. Lexical diversity (MATTR)
11. Numerical precision
12. Title-body coherence (TF-IDF)
13. VADER sentiment polarity
14. Readability (textstat)
15. Named entity density (spaCy NER)
"""

EMOTIONAL_WORDS_NEGATIVE = {
    'outrage', 'outrageous', 'scandalous',
    'disgusting', 'disgrace', 'shameful',
    'evil', 'wicked', 'sinister', 'vile', 'despicable', 'hideous',
    'betrayal', 'betrayed', 'traitor', 'treason', 'liar',
    'bombshell', 'jaw-dropping',
}

EMOTIONAL_WORDS_POSITIVE = {
    'miraculous', 'miracle', 'genius',
}

# Urgency / time pressure - single words matched against tokens, phrases against full text
URGENCY_WORDS = {'hurry', 'quick', 'fast'}
URGENCY_PHRASES = [
    'before it\'s too late', 'last chance', 'limited time',
    'share before', 'must see', 'must read', 'must watch',
    'don\'t miss', 'act now',
]

# Superlatives
SUPERLATIVES = {
    'always', 'everyone', 'no one', 'nobody', 'everybody', 'none',
    'without doubt', 'first ever',
}

VAGUE_ATTRIBUTION_PATTERNS = [
    r'\bsources?\s+say\b',
    r'\bsources?\s+claim\b',
    r'\bsources?\s+report\b',
    r'\bsources?\s+reveal\b',
    r'\bsources?\s+confirm\b',
    r'\bexperts?\s+say\b',
    r'\bexperts?\s+claim\b',
    r'\bexperts?\s+warn\b',
    r'\bscientists?\s+say\b',
    r'\bscientists?\s+claim\b',
    r'\bdoctors?\s+say\b',
    r'\bdoctors?\s+warn\b',
    r'\bsome\s+people\s+say\b',
    r'\bmany\s+believe\b',
    r'\breports?\s+suggest\b',
    r'\breports?\s+indicate\b',
    r'\ballegedly\b',
    r'\breportedly\b',
    r'\baccording\s+to\s+sources\b',
    r'\binsiders?\s+say\b',
    r'\bofficials?\s+say\b',
    r'\bcritics?\s+say\b',
]

CLICKBAIT_PATTERNS = [
    r'you\s+won\'?t\s+believe',
    r'you\'?ll\s+never\s+believe',
    r'you\s+have\s+to\s+see',
    r'then\s+this\s+happened',
    r'and\s+then\s+this',
    r'\bthey\s+don\'?t\s+want\s+you\s+to\s+know\b',
    r'\bwhat\s+they\'?re\s+not\s+telling\s+you\b',
    r'\bthe\s+truth\s+about\b',
    r'\b\d+\s+things?\s+you\b',
    r'\b\d+\s+secrets?\b',
    r'\b\d+\s+shocking\b',
    r'\bmind-?blowing\b',
    r'\bjaw-?dropping\b',
]

CREDIBILITY_INDICATORS = [
    r'according\s+to\s+[A-Z][a-z]+',
    r'said\s+[A-Z][a-z]+\s+[A-Z][a-z]+',
    r'Dr\.\s+[A-Z]',
    r'Prof\.\s+[A-Z]',
    r'Professor\s+[A-Z]',
    r'University\s+of\s+[A-Z]',
    r'published\s+in\s+[A-Z]',
    r'study\s+by\s+[A-Z]',
    r'research\s+(from|by)\s+[A-Z]',
    r'peer-?reviewed',
    r'official\s+statement',
    r'press\s+release',
    r'confirmed\s+by\s+[A-Z]',
    r'spokesperson\s+(for|of)\s+[A-Z]',
]

CONSPIRACY_PHRASES = [
    r'\bwake\s+up\b',
    r'\bopen\s+your\s+eyes\b',
    r'\bsheeple\b',
    r'\bmainstream\s+media\b',
    r'\bmsm\b',
    r'\bcoverup\b',
    r'\bcover[\s-]?up\b',
    r'\bconspiracy\b',
    r'\bthey\s+are\s+lying\b',
    r'\bgovernment\s+doesn\'?t\s+want\b',
    r'\bbig\s+pharma\b',
    r'\bdeep\s+state\b',
    r'\bdo\s+your\s+(own\s+)?research\b',
    r'\bfollow\s+the\s+money\b',
    r'\bconnect\s+the\s+dots\b',
    r'\bask\s+yourself\b',
    r'\bthink\s+about\s+it\b',
    r'\bsilenced\b',
    r'\bcensored\b',
    r'\bsuppressed\b',
    r'\bagenda\b',
    r'\bnarrative\b',
    r'\bpropaganda\b',
]


# Data structures
@dataclass
class LinguisticFeatures:
    word_count: int = 0
    sentence_count: int = 0
    avg_word_length: float = 0.0
    avg_sentence_length: float = 0.0

    clickbait_patterns_found: List[str] = field(default_factory=list)

    emotional_words_ratio: float = 0.0
    emotional_words_found: List[str] = field(default_factory=list)

    exclamation_count: int = 0

    vague_attribution_count: int = 0
    credibility_indicators_count: int = 0
    quote_count: int = 0

    urgency_words_count: int = 0
    superlatives_count: int = 0

    conspiracy_phrases_count: int = 0
    conspiracy_phrases_found: List[str] = field(default_factory=list)
    mattr: float = 0.0
    numerical_precision_count: int = 0
    title_body_coherence: float = 1.0

    vader_compound: float = 0.0
    flesch_reading_ease: float = 0.0

    named_entity_density: float = 0.0
    named_entity_types: Dict[str, int] = field(default_factory=dict)

    headline_consistency: Optional[float] = None  # NLI entailment score (0=inconsistent, 1=consistent)

    flags: List[dict] = field(default_factory=list)


@dataclass
class AnalysisResult:
    score: float  # 0.0 (fake) – 1.0 (reliable)
    flags: List[dict] = field(default_factory=list)
    explanation: Dict = field(default_factory=dict)
    headline_consistency: Optional[float] = None



# Analyzer
class LinguisticAnalyzer:

    def __init__(self, nlp=None):
        if nlp is not None:
            self.nlp = nlp
        else:
            try:
                self.nlp = spacy.load("en_core_web_sm", disable=["lemmatizer"])
            except OSError:
                from spacy.cli import download
                download("en_core_web_sm")
                self.nlp = spacy.load("en_core_web_sm", disable=["lemmatizer"])
            self.nlp.max_length = 1_500_000

        self.vader = SentimentIntensityAnalyzer()

        # NLI for headline-body consistency
        self._nli = None
        try:
            from transformers import pipeline as hf_pipeline
            self._nli = hf_pipeline(
                "text-classification",
                model="cross-encoder/nli-deberta-v3-small",
                top_k=None,
                device=-1,
            )
            print("[LinguisticAnalyzer] NLI model loaded.")
        except Exception as e:
            warnings.warn(f"[LinguisticAnalyzer] NLI model unavailable — headline consistency disabled. ({e})")

        self.clickbait_patterns = [re.compile(p, re.IGNORECASE) for p in CLICKBAIT_PATTERNS]
        self.vague_patterns = [re.compile(p, re.IGNORECASE) for p in VAGUE_ATTRIBUTION_PATTERNS]
        self.credibility_patterns = [re.compile(p) for p in CREDIBILITY_INDICATORS]
        self.conspiracy_patterns = [re.compile(p, re.IGNORECASE) for p in CONSPIRACY_PHRASES]

        self.emotional_words = EMOTIONAL_WORDS_NEGATIVE | EMOTIONAL_WORDS_POSITIVE
        self.urgency_words = URGENCY_WORDS
        self.urgency_phrases = URGENCY_PHRASES
        self.superlatives = SUPERLATIVES

    def _count_matches(self, patterns: list, text: str) -> int:
        return sum(len(p.findall(text)) for p in patterns)

    def _compute_mattr(self, tokens: List[str], window_size: int = 50) -> float:
        # Moving Average Type-Token Ratio
        if not tokens:
            return 0.0
        if len(tokens) < window_size:
            return len(set(tokens)) / len(tokens)
        ttr_values = [
            len(set(tokens[i:i + window_size])) / window_size
            for i in range(len(tokens) - window_size + 1)
        ]
        return sum(ttr_values) / len(ttr_values)

    def extract_features(self, text: str, title: str = "") -> LinguisticFeatures:
        features = LinguisticFeatures()

        if not text:
            return features

        full_text = f"{title} {text}" if title else text
        text_lower = full_text.lower()

        doc = self.nlp(full_text)
        tokens = [token for token in doc if token.is_alpha]
        words = [token.text for token in tokens]
        words_lower = [token.lower_ for token in tokens]
        sentences = list(doc.sents)

        features.word_count = len(words)
        features.sentence_count = len(sentences)
        if sentences:
            features.avg_sentence_length = features.word_count / len(sentences)

        analysis_text = title.lower() if title else text_lower[:500]
        for pattern in self.clickbait_patterns:
            features.clickbait_patterns_found.extend(pattern.findall(analysis_text))
        features.clickbait_score = min(1.0, len(features.clickbait_patterns_found) * 0.25)

        emotional_found = [w for w in words_lower if w in self.emotional_words]
        features.emotional_words_found = list(set(emotional_found))[:10]
        if words:
            features.emotional_words_ratio = len(emotional_found) / len(words)

        features.exclamation_count = full_text.count('!')

        features.vague_attribution_count = self._count_matches(self.vague_patterns, text_lower)
        features.credibility_indicators_count = self._count_matches(self.credibility_patterns, full_text)

        quotes = re.findall(r'["\u201C\u201D]([^"\u201C\u201D]{10,})["\u201C\u201D]', full_text)
        features.quote_count = len(quotes)

        features.urgency_words_count = (
            sum(1 for w in words_lower if w in self.urgency_words) +
            sum(1 for p in self.urgency_phrases if p in text_lower)
        )

        superlatives_found = [w for w in words_lower if w in self.superlatives]
        pos_superlatives = [token.text for token in tokens
                            if token.tag_ in ("JJS", "RBS")
                            and token.lower_ not in self.superlatives]
        features.superlatives_count = len(superlatives_found) + len(pos_superlatives)

        for pattern in self.conspiracy_patterns:
            matches = pattern.findall(text_lower)
            features.conspiracy_phrases_count += len(matches)
            features.conspiracy_phrases_found.extend(matches)
        features.conspiracy_phrases_found = list(set(features.conspiracy_phrases_found))[:10]

        if words_lower:
            features.mattr = self._compute_mattr(words_lower)

        features.numerical_precision_count = len(re.findall(
            r'\b\d+\.?\d*\s*%|\b\d{1,3}(?:,\d{3})+\b|\$\d|\b\d{4}\b|\b\d+\.\d+\b',
            full_text
        ))

        if title and words:
            try:
                tfidf = TfidfVectorizer(stop_words='english')
                vectors = tfidf.fit_transform([title, full_text])
                features.title_body_coherence = round(
                    float(cosine_similarity(vectors[0:1], vectors[1:2])[0][0]), 2
                )
            except ValueError:
                features.title_body_coherence = 1.0

        features.vader_compound = self.vader.polarity_scores(full_text)["compound"]

        if features.word_count >= 10:
            features.flesch_reading_ease = textstat.flesch_reading_ease(full_text)

        if len(doc) > 0:
            features.named_entity_density = len(doc.ents) / len(doc)
            features.named_entity_types = dict(Counter(ent.label_ for ent in doc.ents))

        # NLI headline-body consistency
        if title and len(text) >= 50 and self._nli is not None:
            try:
                # premise = first ~500 chars of body; hypothesis = headline
                result = self._nli({"text": text[:500], "text_pair": title})
                scores = {item["label"].lower(): item["score"] for item in result}
                features.headline_consistency = round(scores.get("entailment", 0.0), 3)
            except Exception:
                pass

        features.flags = self._generate_flags(features, title)

        return features

    def _generate_flags(self, f: LinguisticFeatures, title: str) -> List[dict]:
        flags = []

        if f.clickbait_patterns_found:
            flags.append({"code": "CLICKBAIT", "description": f"Clickbait patterns: {', '.join(f.clickbait_patterns_found[:3])}", "positive": False})
        if f.emotional_words_ratio > 0.05:
            flags.append({"code": "HIGH_EMOTION", "description": f"{f.emotional_words_ratio:.1%} emotional words", "positive": False})
        if f.exclamation_count > 3:
            flags.append({"code": "EXCLAMATION_OVERUSE", "description": f"{f.exclamation_count} exclamation marks", "positive": False})
        if f.vague_attribution_count > 2 and f.credibility_indicators_count == 0:
            flags.append({"code": "VAGUE_SOURCES", "description": "No named sources — uses vague attribution", "positive": False})
        if f.urgency_words_count > 2:
            flags.append({"code": "URGENCY", "description": f"{f.urgency_words_count} urgency words", "positive": False})

        superlative_ratio = f.superlatives_count / max(1, f.word_count)
        if superlative_ratio > 0.02:
            flags.append({"code": "SUPERLATIVES", "description": f"{f.superlatives_count} absolute terms ({superlative_ratio:.1%} of words)", "positive": False})
        if f.word_count > 300 and f.quote_count == 0:
            flags.append({"code": "NO_QUOTES", "description": "No direct quotes in a long article", "positive": False})
        if f.conspiracy_phrases_count > 0:
            flags.append({"code": "CONSPIRACY", "description": f"Conspiracy language: {', '.join(f.conspiracy_phrases_found[:3])}", "positive": False})
        if f.word_count > 100 and f.mattr < 0.4:
            flags.append({"code": "LOW_DIVERSITY", "description": f"Repetitive vocabulary ({f.mattr:.1%} unique words)", "positive": False})
        if f.title_body_coherence < 0.05 and title:
            flags.append({"code": "TITLE_DISCONNECT", "description": f"Title is disconnected from body content (similarity: {f.title_body_coherence:.2f})", "positive": False})
        if f.word_count > 200 and f.named_entity_density < 0.01:
            flags.append({"code": "LOW_ENTITIES", "description": "Very few named entities — lacks specifics", "positive": False})
        if f.vader_compound < -0.7 and f.named_entity_density < 0.02 and f.quote_count == 0:
            flags.append({"code": "EMOTIONAL_RANT", "description": f"Extremely negative tone (VADER: {f.vader_compound:.2f}) with no entities or quotes", "positive": False})
        if f.flesch_reading_ease > 80 and f.word_count > 200:
            flags.append({"code": "VERY_SIMPLE_LANGUAGE", "description": "Written at a very simple reading level", "positive": False})
        if f.headline_consistency is not None and f.headline_consistency < 0.25:
            flags.append({"code": "HEADLINE_INCONSISTENCY", "description": f"Headline is not supported by article body (NLI entailment: {f.headline_consistency:.2f})", "positive": False})

        return flags

    def analyze(self, text: str, title: str = "") -> AnalysisResult:
        f = self.extract_features(text, title)

        score = 0.5

        # Penalties
        score -= min(0.12, len(f.clickbait_patterns_found) * 0.06)

        if f.emotional_words_ratio > 0.03:
            score -= min(0.08, (f.emotional_words_ratio - 0.03) * 2.5)

        if f.exclamation_count > 2:
            score -= min(0.05, (f.exclamation_count - 2) * 0.015)

        if f.vague_attribution_count > 0:
            score -= min(0.05, f.vague_attribution_count * 0.02)

        if f.urgency_words_count > 2:
            score -= min(0.04, (f.urgency_words_count - 2) * 0.02)

        superlative_ratio = f.superlatives_count / max(1, f.word_count)
        if superlative_ratio > 0.02:
            score -= min(0.05, (superlative_ratio - 0.02) * 2.5)

        if f.conspiracy_phrases_count > 0:
            score -= min(0.10, f.conspiracy_phrases_count * 0.04)

        if f.word_count > 100 and f.mattr < 0.45:
            score -= min(0.04, (0.45 - f.mattr) * 0.3)

        if f.title_body_coherence < 0.05 and title:
            score -= 0.05

        if f.vader_compound < -0.7 and f.named_entity_density < 0.02 and f.quote_count == 0:
            score -= min(0.05, abs(f.vader_compound + 0.7) * 0.17)

        if f.headline_consistency is not None:
            if f.headline_consistency < 0.20:
                score -= 0.08
            elif f.headline_consistency < 0.35:
                score -= 0.04
            elif f.headline_consistency > 0.65:
                score += 0.05

        # Bonuses
        if f.credibility_indicators_count > 0:
            score += min(0.15, f.credibility_indicators_count * 0.04)

        if f.quote_count > 0:
            score += min(0.12, f.quote_count * 0.03)

        if f.numerical_precision_count > 0:
            score += min(0.10, f.numerical_precision_count * 0.025)

        if f.word_count > 500 and 15 < f.avg_sentence_length < 30:
            score += 0.08

        if f.named_entity_density > 0.02:
            score += min(0.12, (f.named_entity_density - 0.02) * 2)

        if 40 <= f.flesch_reading_ease <= 70:
            score += 0.08

        score = round(max(0.0, min(1.0, score)), 3)

        return AnalysisResult(
            score=score,
            flags=f.flags,
            explanation=self._generate_explanation(f, score),
            headline_consistency=f.headline_consistency,
        )

    def _generate_explanation(self, f: LinguisticFeatures, score: float) -> dict:
        if score >= 0.7:
            verdict = "Text shows characteristics of reliable journalism"
            verdict_level = "RELIABLE"
        elif score >= 0.5:
            verdict = "Text has some concerning features but is not clearly unreliable"
            verdict_level = "MIXED"
        elif score >= 0.3:
            verdict = "Text shows multiple characteristics associated with unreliable content"
            verdict_level = "CONCERNING"
        else:
            verdict = "Text shows strong characteristics of fake news or clickbait"
            verdict_level = "UNRELIABLE"

        signals = []

        if f.clickbait_patterns_found:
            signals.append({"label": "Clickbait patterns", "value": ", ".join(f.clickbait_patterns_found[:3]), "positive": False})
        if f.emotional_words_ratio > 0.05:
            signals.append({"label": "Emotional language", "value": f"{f.emotional_words_ratio:.1%} of words", "positive": False})
        if f.vague_attribution_count > 2:
            signals.append({"label": "Vague source attribution", "value": f"{f.vague_attribution_count} instances", "positive": False})
        if f.credibility_indicators_count > 0:
            signals.append({"label": "Credibility indicators", "value": str(f.credibility_indicators_count), "positive": True})
        if f.quote_count > 0:
            signals.append({"label": "Direct quotes", "value": str(f.quote_count), "positive": True})
        if f.conspiracy_phrases_count > 0:
            signals.append({"label": "Conspiracy language", "value": ", ".join(f.conspiracy_phrases_found[:3]), "positive": False})
        if f.numerical_precision_count > 2:
            signals.append({"label": "Specific data points", "value": f"{f.numerical_precision_count} found", "positive": True})
        if f.named_entity_density > 0.03:
            signals.append({"label": "Named entities", "value": f"{len(f.named_entity_types)} types", "positive": True})
        elif f.word_count > 200 and f.named_entity_density < 0.01:
            signals.append({"label": "Named entities", "value": "very few", "positive": False})

        return {"verdict": verdict, "verdict_level": verdict_level, "signals": signals}