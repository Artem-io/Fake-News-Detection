"""
Linguistic Analyzer for Fake News Detection
============================================

Moduł analizy heurystycznej tekstu pod kątem cech typowych dla fake news.
Bazuje na badaniach naukowych, sprawdzonych wzorcach oraz NLP (spaCy, VADER, textstat).

Główne cechy analizowane:
1. Clickbait patterns (nagłówki)
2. Emotional/sensational language
3. ALL CAPS abuse
4. Exclamation marks overuse
5. Vague attribution ("sources say")
6. Urgency/fear words
7. Superlatives and absolutes (lexicon + POS tags JJS/RBS)
8. Question headlines (Betteridge's Law)
9. First person pronouns (nietypowe dla newsów)
10. Credential indicators (cytaty, źródła)
11. Conspiracy/propaganda language
12. Hedging words (wskaźnik rzetelności)
13. Negation density
14. Lexical diversity (MATTR)
15. Numerical precision
16. Title-body coherence
17. VADER sentiment polarity
18. Readability (textstat)
19. Passive voice detection (spaCy dep parse)
20. Named entity density (spaCy NER)
21. Opening sentence sentiment profiling

Źródła naukowe:
- Horne & Adali (2017) "This Just In: Fake News Packs a Lot in Title"
- Rashkin et al. (2017) "Truth of Varying Shades"
- Pérez-Rosas et al. (2018) "Automatic Detection of Fake News"
"""

import re
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional
from collections import Counter
import math

import spacy
import textstat
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# =============================================================================
# LEKSYKONY I WZORCE
# =============================================================================

# Słowa manipulacyjne — język celowo wyolbrzymiający, nie fakty
# NIE zawiera słów które są standardowym słownictwem newsowym (killed, attack,
# crisis, death, etc.) — te są normalne w relacjach o wojnach, klęskach, kryminałach.
EMOTIONAL_WORDS_NEGATIVE = {
    # Manipulacja / wyolbrzymienie (NIE fakty)
    'horror', 'horrifying', 'nightmare', 'catastrophic',
    'panic', 'afraid', 'scared', 'alarming',

    # Złość / oburzenie — język opinii, nie relacji
    'outrage', 'outrageous', 'scandal', 'scandalous', 'shocking', 'shocked',
    'disgusting', 'disgrace', 'shameful', 'shame',
    'evil', 'wicked', 'sinister', 'vile', 'despicable', 'hideous',
    'betrayal', 'betrayed', 'traitor', 'treason', 'liar', 'lies', 'lying',

    # Sensacja
    'explosive', 'bombshell', 'stunning', 'jaw-dropping', 'unbelievable',
    'incredible', 'insane', 'crazy', 'wild',
}

# Słowa emocjonalne (pozytywne - też mogą być clickbait)
EMOTIONAL_WORDS_POSITIVE = {
    'amazing', 'awesome', 'brilliant', 'fantastic', 'wonderful', 'incredible',
    'unbelievable', 'miraculous', 'miracle',
    'genius', 'perfect', 'ultimate', 'epic',
}

# Słowa pilności / presji czasowej
# NIE zawiera 'breaking', 'now', 'today', 'developing' — to standardowe słownictwo newsowe
URGENCY_WORDS = {
    'hurry', 'quick', 'fast',
    'before it\'s too late', 'last chance', 'limited time',
    'share before', 'must see', 'must read', 'must watch',
    'don\'t miss', 'act now',
}

# Superlatywy i absoluty
# Usunięto 'all', 'every', 'most', 'only', 'never' — zbyt częste w normalnych artykułach
SUPERLATIVES = {
    'always', 'everyone', 'no one', 'nobody', 'everybody',
    'none', 'completely', 'totally', 'absolutely',
    'definitely', 'certainly', 'undoubtedly', 'without doubt',
    'worst', 'greatest', 'biggest', 'smallest',
    'first ever', 'unprecedented',
}

# Niejasne źródła (vague attribution)
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
    r'\bofficials?\s+say\b',  # bez nazwiska
    r'\bcritics?\s+say\b',
]

# Wzorce clickbait w tytułach
CLICKBAIT_PATTERNS = [
    # "You won't believe..."
    r'you\s+won\'?t\s+believe',
    r'you\'?ll\s+never\s+believe',
    r'you\s+have\s+to\s+see',
    r'you\s+need\s+to\s+know',
    r'you\s+should\s+know',

    # "What happens next..."
    r'what\s+happens?\s+next',
    r'what\s+happened\s+next',
    r'then\s+this\s+happened',
    r'and\s+then\s+this',

    # "This is why..."
    r'this\s+is\s+why',
    r'here\'?s\s+why',
    r'the\s+reason\s+why',

    # "Secret/Hidden..."
    r'\bsecret\b',
    r'\bhidden\b',
    r'\bthey\s+don\'?t\s+want\s+you\s+to\s+know\b',
    r'\bwhat\s+they\'?re\s+not\s+telling\s+you\b',
    r'\bthe\s+truth\s+about\b',

    # Numbers listicles
    r'\b\d+\s+reasons?\s+why\b',
    r'\b\d+\s+things?\s+you\b',
    r'\b\d+\s+ways?\s+to\b',
    r'\b\d+\s+secrets?\b',
    r'\b\d+\s+shocking\b',

    # Emotional hooks
    r'\bshocking\b',
    r'\bunbelievable\b',
    r'\bmind-?blowing\b',
    r'\bjaw-?dropping\b',
    r'\bheart-?breaking\b',
    r'\blife-?changing\b',

    # Questions (Betteridge's Law)
    r'\bis\s+this\s+the\s+end\b',
    r'\bcould\s+this\s+be\b',
    r'\bwill\s+this\s+change\b',
]

# Wskaźniki wiarygodności (pozytywne)
CREDIBILITY_INDICATORS = [
    r'according\s+to\s+[A-Z][a-z]+',  # According to [Name]
    r'said\s+[A-Z][a-z]+\s+[A-Z][a-z]+',  # said John Smith
    r'Dr\.\s+[A-Z]',  # Dr. [Name]
    r'Prof\.\s+[A-Z]',  # Prof. [Name]
    r'Professor\s+[A-Z]',
    r'University\s+of\s+[A-Z]',
    r'published\s+in\s+[A-Z]',  # published in [Journal]
    r'study\s+by\s+[A-Z]',
    r'research\s+(from|by)\s+[A-Z]',
    r'peer-?reviewed',
    r'official\s+statement',
    r'press\s+release',
    r'confirmed\s+by\s+[A-Z]',
    r'spokesperson\s+(for|of)\s+[A-Z]',
]

# Zaimki pierwszej osoby (nietypowe dla rzetelnych newsów)
FIRST_PERSON_PRONOUNS = {'i', 'me', 'my', 'mine', 'myself', 'we', 'us', 'our', 'ours'}

# Język konspiracyjny / propagandowy
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

# Słowa hedgingowe (wskaźnik rzetelności)
HEDGING_WORDS = {
    'may', 'might', 'could', 'possibly', 'perhaps', 'likely', 'unlikely',
    'suggests', 'indicates', 'appears', 'seems', 'estimated', 'approximately',
    'roughly', 'preliminary', 'potential', 'probable', 'arguably',
}

# Negacje (nadużywane w fake news)
NEGATION_WORDS = {
    'not', 'no', 'never', 'nothing', 'nowhere', 'neither', 'nor',
    'nobody', 'none',
}


# =============================================================================
# STRUKTURY DANYCH
# =============================================================================

@dataclass
class LinguisticFeatures:
    """Wyekstrahowane cechy lingwistyczne"""

    # Podstawowe metryki
    word_count: int = 0
    sentence_count: int = 0
    avg_word_length: float = 0.0
    avg_sentence_length: float = 0.0

    # Clickbait
    clickbait_patterns_found: List[str] = field(default_factory=list)
    clickbait_score: float = 0.0

    # Emocjonalność
    emotional_words_count: int = 0
    emotional_words_ratio: float = 0.0
    emotional_words_found: List[str] = field(default_factory=list)

    # Formatowanie
    caps_ratio: float = 0.0
    exclamation_count: int = 0
    question_mark_count: int = 0

    # Źródła
    vague_attribution_count: int = 0
    credibility_indicators_count: int = 0
    quote_count: int = 0

    # Inne
    urgency_words_count: int = 0
    superlatives_count: int = 0
    first_person_ratio: float = 0.0

    # Conspiracy / hedging / negation / lexical
    conspiracy_phrases_count: int = 0
    conspiracy_phrases_found: List[str] = field(default_factory=list)
    hedging_words_count: int = 0
    hedging_ratio: float = 0.0
    negation_ratio: float = 0.0
    lexical_diversity: float = 0.0  # MATTR
    mattr: float = 0.0
    numerical_precision_count: int = 0
    title_body_coherence: float = 1.0

    # VADER sentiment
    vader_compound: float = 0.0
    vader_positive: float = 0.0
    vader_negative: float = 0.0
    vader_neutral: float = 0.0

    # Readability (textstat)
    flesch_reading_ease: float = 0.0
    flesch_kincaid_grade: float = 0.0
    smog_index: float = 0.0
    coleman_liau_index: float = 0.0

    # spaCy-derived
    passive_voice_ratio: float = 0.0
    named_entity_density: float = 0.0
    named_entity_types: Dict[str, int] = field(default_factory=dict)

    # Opening sentence analysis
    opening_sentences_sentiment: List[Dict] = field(default_factory=list)

    # Flagi problemów
    flags: List[str] = field(default_factory=list)


@dataclass
class AnalysisResult:
    """Wynik analizy lingwistycznej"""

    # Główny wynik
    score: float  # 0.0 (fake) - 1.0 (reliable)
    confidence: float  # Pewność oceny

    # Szczegóły
    features: LinguisticFeatures = None
    flags: List[str] = field(default_factory=list)

    # Wyjaśnienie
    explanation: str = ""
    details: Dict = field(default_factory=dict)


# =============================================================================
# GŁÓWNA KLASA ANALIZATORA
# =============================================================================

class LinguisticAnalyzer:
    """
    Analizator lingwistyczny tekstu pod kątem fake news.

    Używa heurystyk bazujących na badaniach naukowych + spaCy, VADER, textstat.
    """

    def __init__(self, nlp=None):
        # spaCy model — accept shared instance or load own
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

        # VADER sentiment analyzer
        self.vader = SentimentIntensityAnalyzer()

        # Kompiluj wzorce regex dla wydajności
        self.clickbait_patterns = [
            re.compile(p, re.IGNORECASE) for p in CLICKBAIT_PATTERNS
        ]
        self.vague_patterns = [
            re.compile(p, re.IGNORECASE) for p in VAGUE_ATTRIBUTION_PATTERNS
        ]
        self.credibility_patterns = [
            re.compile(p) for p in CREDIBILITY_INDICATORS
        ]
        self.conspiracy_patterns = [
            re.compile(p, re.IGNORECASE) for p in CONSPIRACY_PHRASES
        ]

        # Połącz leksykony
        self.emotional_words = EMOTIONAL_WORDS_NEGATIVE | EMOTIONAL_WORDS_POSITIVE
        self.urgency_words = URGENCY_WORDS
        self.superlatives = SUPERLATIVES
        self.hedging_words = HEDGING_WORDS
        self.negation_words = NEGATION_WORDS

    # =========================================================================
    # METODY POMOCNICZE NLP
    # =========================================================================

    def _compute_mattr(self, tokens: List[str], window_size: int = 50) -> float:
        """Moving Average Type-Token Ratio — stabilniejszy niż TTR dla różnych długości."""
        if not tokens:
            return 0.0
        if len(tokens) < window_size:
            return len(set(tokens)) / len(tokens)
        ttr_values = []
        for i in range(len(tokens) - window_size + 1):
            window = tokens[i : i + window_size]
            ttr_values.append(len(set(window)) / window_size)
        return sum(ttr_values) / len(ttr_values)

    def _detect_passive_voice(self, doc) -> float:
        """Wykryj stronę bierną przez spaCy dependency labels (nsubjpass / auxpass)."""
        passive_count = 0
        clause_count = 0
        for token in doc:
            if token.dep_ in ("ROOT", "conj") and token.pos_ == "VERB":
                clause_count += 1
            if token.dep_ == "nsubjpass":
                passive_count += 1
        if clause_count == 0:
            return 0.0
        return passive_count / clause_count

    def _analyze_opening_sentences(self, doc, n: int = 3) -> List[Dict]:
        """VADER sentiment na pierwszych n zdaniach — profiluje ton otwarcia."""
        results = []
        for sent in list(doc.sents)[:n]:
            scores = self.vader.polarity_scores(sent.text)
            results.append({
                "text": sent.text.strip(),
                "compound": scores["compound"],
                "pos": scores["pos"],
                "neg": scores["neg"],
                "neu": scores["neu"],
            })
        return results

    # =========================================================================
    # EKSTRAKCJA CECH
    # =========================================================================

    def extract_features(self, text: str, title: str = "") -> LinguisticFeatures:
        """
        Wyekstrahuj wszystkie cechy lingwistyczne z tekstu.

        Args:
            text: Treść artykułu
            title: Tytuł artykułu (opcjonalnie)

        Returns:
            LinguisticFeatures z wszystkimi metrykami
        """
        features = LinguisticFeatures()

        if not text:
            return features

        # Połącz tytuł i tekst do analizy
        full_text = f"{title} {text}" if title else text
        text_lower = full_text.lower()

        # spaCy tokenizacja
        doc = self.nlp(full_text)

        tokens = [token for token in doc if token.is_alpha]
        words = [token.text for token in tokens]
        words_lower = [token.lower_ for token in tokens]
        sentences = list(doc.sents)

        # 1. Podstawowe metryki
        features.word_count = len(words)
        features.sentence_count = len(sentences)

        if words:
            features.avg_word_length = sum(len(w) for w in words) / len(words)
        if sentences:
            features.avg_sentence_length = features.word_count / len(sentences)

        # 2. Clickbait (szczególnie w tytule)
        analysis_text = title.lower() if title else text_lower[:500]
        for pattern in self.clickbait_patterns:
            matches = pattern.findall(analysis_text)
            features.clickbait_patterns_found.extend(matches)

        features.clickbait_score = min(1.0, len(features.clickbait_patterns_found) * 0.25)

        # 3. Słowa emocjonalne
        emotional_found = [w for w in words_lower if w in self.emotional_words]
        features.emotional_words_count = len(emotional_found)
        features.emotional_words_found = list(set(emotional_found))[:10]

        if words:
            features.emotional_words_ratio = len(emotional_found) / len(words)

        # 4. Wykrzykniki i pytajniki
        features.exclamation_count = full_text.count('!')
        features.question_mark_count = full_text.count('?')

        # 6. Niejasne źródła
        for pattern in self.vague_patterns:
            features.vague_attribution_count += len(pattern.findall(text_lower))

        # 7. Wskaźniki wiarygodności
        for pattern in self.credibility_patterns:
            features.credibility_indicators_count += len(pattern.findall(full_text))

        # 8. Cytaty (tekst w cudzysłowach)
        quotes = re.findall(r'["\u201C\u201D]([^"\u201C\u201D]{10,})["\u201C\u201D]', full_text)
        features.quote_count = len(quotes)

        # 9. Słowa pilności
        urgency_found = [w for w in words_lower if w in self.urgency_words]
        features.urgency_words_count = len(urgency_found)

        for phrase in ['share before', 'must see', 'act now', 'last chance']:
            if phrase in text_lower:
                features.urgency_words_count += 1

        # 10. Superlatywy (lexicon + POS tags JJS/RBS)
        superlatives_found = [w for w in words_lower if w in self.superlatives]
        pos_superlatives = [token.text for token in tokens
                           if token.tag_ in ("JJS", "RBS")
                           and token.lower_ not in self.superlatives]
        features.superlatives_count = len(superlatives_found) + len(pos_superlatives)

        # 11. Zaimki pierwszej osoby
        first_person_count = sum(1 for w in words_lower if w in FIRST_PERSON_PRONOUNS)
        if words:
            features.first_person_ratio = first_person_count / len(words)

        # 12. Język konspiracyjny / propagandowy
        for pattern in self.conspiracy_patterns:
            matches = pattern.findall(text_lower)
            if matches:
                features.conspiracy_phrases_count += len(matches)
                features.conspiracy_phrases_found.extend(matches)
        features.conspiracy_phrases_found = list(set(features.conspiracy_phrases_found))[:10]

        # 13. Hedging (wskaźnik rzetelności)
        hedging_found = [w for w in words_lower if w in self.hedging_words]
        features.hedging_words_count = len(hedging_found)
        if words:
            features.hedging_ratio = len(hedging_found) / len(words)

        # 14. Negacje (spaCy handles contractions: "don't" -> "do" + "n't")
        negation_count = 0
        for token in tokens:
            if token.lower_ in self.negation_words:
                negation_count += 1
            elif token.dep_ == "neg":
                negation_count += 1
        if words:
            features.negation_ratio = negation_count / len(words)

        # 15. Lexical diversity (MATTR zamiast TTR)
        if words_lower:
            features.mattr = self._compute_mattr(words_lower)
            features.lexical_diversity = features.mattr

        # 16. Numerical precision (konkretne dane = wiarygodność)
        features.numerical_precision_count = len(re.findall(
            r'\b\d+\.?\d*\s*%|\b\d{1,3}(?:,\d{3})+\b|\$\d|\b\d{4}\b|\b\d+\.\d+\b',
            full_text
        ))

        # 17. Title-body coherence (TF-IDF cosine similarity)
        if title and words:
            try:
                tfidf = TfidfVectorizer(stop_words='english')
                vectors = tfidf.fit_transform([title, full_text])
                similarity = cosine_similarity(vectors[0:1], vectors[1:2])[0][0]
                features.title_body_coherence = round(float(similarity), 2)
            except ValueError:
                features.title_body_coherence = 1.0

        # 18. VADER sentiment
        vader_scores = self.vader.polarity_scores(full_text)
        features.vader_compound = vader_scores["compound"]
        features.vader_positive = vader_scores["pos"]
        features.vader_negative = vader_scores["neg"]
        features.vader_neutral = vader_scores["neu"]

        # 19. Readability (textstat)
        if features.word_count >= 10:
            features.flesch_reading_ease = textstat.flesch_reading_ease(full_text)
            features.flesch_kincaid_grade = textstat.flesch_kincaid_grade(full_text)
            features.smog_index = textstat.smog_index(full_text)
            features.coleman_liau_index = textstat.coleman_liau_index(full_text)

        # 20. Passive voice (spaCy dep parse)

        # 21. Named entity density (spaCy NER)
        if len(doc) > 0:
            features.named_entity_density = len(doc.ents) / len(doc)
            ent_counter = Counter(ent.label_ for ent in doc.ents)
            features.named_entity_types = dict(ent_counter)

        # 22. Opening sentence sentiment

        # Generuj flagi
        features.flags = self._generate_flags(features, title)

        return features

    def _generate_flags(self, f: LinguisticFeatures, title: str) -> List[str]:
        """Wygeneruj listę flag/ostrzeżeń"""
        flags = []

        # Clickbait
        if f.clickbait_patterns_found:
            flags.append(f"CLICKBAIT: {', '.join(f.clickbait_patterns_found[:3])}")

        # Emocjonalność
        if f.emotional_words_ratio > 0.05:
            flags.append(f"HIGH_EMOTION: {f.emotional_words_ratio:.1%} emotional words")

        # Wykrzykniki
        if f.exclamation_count > 3:
            flags.append(f"EXCLAMATION_OVERUSE: {f.exclamation_count} exclamation marks")

        # Niejasne źródła
        if f.vague_attribution_count > 2 and f.credibility_indicators_count == 0:
            flags.append("VAGUE_SOURCES: no named sources")

        # Pilność (only manipulative urgency)
        if f.urgency_words_count > 2:
            flags.append(f"URGENCY: {f.urgency_words_count} urgency words")

        # Superlatywy (ratio-based)
        superlative_ratio = f.superlatives_count / max(1, f.word_count)
        if superlative_ratio > 0.02:
            flags.append(f"SUPERLATIVES: {f.superlatives_count} absolute terms ({superlative_ratio:.1%} of words)")

        # Pytanie w tytule (Betteridge's Law)
        if title and title.strip().endswith('?'):
            flags.append("QUESTION_HEADLINE: headline is a question")

        # Pierwsza osoba
        if f.first_person_ratio > 0.02:
            flags.append(f"FIRST_PERSON: {f.first_person_ratio:.1%} first-person pronouns")

        # Brak cytatów w długim tekście
        if f.word_count > 300 and f.quote_count == 0:
            flags.append("NO_QUOTES: no direct quotes in article")

        # Język konspiracyjny
        if f.conspiracy_phrases_count > 0:
            flags.append(f"CONSPIRACY: {', '.join(f.conspiracy_phrases_found[:3])}")


        # Niska różnorodność leksykalna (MATTR)
        if f.word_count > 100 and f.lexical_diversity < 0.4:
            flags.append(f"LOW_DIVERSITY: {f.lexical_diversity:.1%} unique words")

        # Brak hedgingu (zbyt definitive)
        if f.word_count > 200 and f.hedging_ratio == 0:
            flags.append("NO_HEDGING: no qualifying language")

        # Title-body disconnect
        if f.title_body_coherence < 0.05 and title:
            flags.append(f"TITLE_DISCONNECT: TF-IDF similarity {f.title_body_coherence:.2f}")

        # Bardzo niska gęstość encji (brak konkretów)
        if f.word_count > 200 and f.named_entity_density < 0.01:
            flags.append("LOW_ENTITIES: very few named entities")

        # Silnie negatywny VADER BEZ faktów (emotional rant)
        if (f.vader_compound < -0.7
                and f.named_entity_density < 0.02
                and f.quote_count == 0):
            flags.append(f"EMOTIONAL_RANT: VADER={f.vader_compound:.2f} with no entities/quotes")

        # Zbyt prosty język w długim tekście
        if f.flesch_reading_ease > 80 and f.word_count > 200:
            flags.append("VERY_SIMPLE_LANGUAGE: Flesch score > 80")

        return flags

    # =========================================================================
    # OBLICZANIE WYNIKU
    # =========================================================================

    def analyze(self, text: str, title: str = "") -> AnalysisResult:
        """
        Przeprowadź pełną analizę lingwistyczną.

        Args:
            text: Treść artykułu
            title: Tytuł artykułu (opcjonalnie)

        Returns:
            AnalysisResult z oceną i szczegółami
        """
        features = self.extract_features(text, title)
        f = features  # shorthand

        # Oblicz score (0.0 = fake, 1.0 = reliable)
        score = 0.5  # Punkt startowy
        decisions = []  # Per-submodule decisions

        # =====================================================================
        # BALANCED SCORING
        # Max penalties: ~-0.80 | Max bonuses: ~+0.78
        # From baseline 0.5 → range [0.0, 1.0] is reachable from both sides
        # =====================================================================

        # === KARY (obniżają score) — total budget: ~0.80 ===

        # 1. Clickbait (-0.12 max)
        penalty = min(0.12, len(f.clickbait_patterns_found) * 0.06)
        score -= penalty
        decisions.append({
            "feature": "clickbait",
            "value": f.clickbait_patterns_found,
            "score_impact": round(-penalty, 3) if penalty else 0.0,
            "triggered": bool(f.clickbait_patterns_found),
            "reason": f"Found {len(f.clickbait_patterns_found)} clickbait pattern(s): {', '.join(f.clickbait_patterns_found[:5])}"
                if f.clickbait_patterns_found else "No clickbait patterns detected.",
            "threshold": "any match",
        })

        # 2. Emocjonalność (-0.08 max)
        penalty = 0.0
        if f.emotional_words_ratio > 0.03:
            penalty = min(0.08, (f.emotional_words_ratio - 0.03) * 2.5)
            score -= penalty
        decisions.append({
            "feature": "emotional_language",
            "value": {"ratio": round(f.emotional_words_ratio, 4), "count": f.emotional_words_count, "words": f.emotional_words_found},
            "score_impact": round(-penalty, 3),
            "triggered": penalty > 0,
            "reason": f"{f.emotional_words_ratio:.1%} of words are emotional ({f.emotional_words_count} found: {', '.join(f.emotional_words_found[:5])}). Threshold: >3%."
                if penalty > 0 else f"Emotional word ratio ({f.emotional_words_ratio:.1%}) is within normal range (<=3%).",
            "threshold": ">3%",
        })

        # 3. Wykrzykniki (-0.05 max)
        penalty = 0.0
        if f.exclamation_count > 2:
            penalty = min(0.05, (f.exclamation_count - 2) * 0.015)
            score -= penalty
        decisions.append({
            "feature": "exclamation_marks",
            "value": f.exclamation_count,
            "score_impact": round(-penalty, 3),
            "triggered": penalty > 0,
            "reason": f"{f.exclamation_count} exclamation marks found. Professional journalism rarely uses more than 2."
                if penalty > 0 else f"{f.exclamation_count} exclamation mark(s) — within normal range (<=2).",
            "threshold": ">2",
        })

        # 5. Niejasne źródła (-0.05 max)
        penalty = 0.0
        if f.vague_attribution_count > 0:
            penalty = min(0.05, f.vague_attribution_count * 0.02)
            score -= penalty
        decisions.append({
            "feature": "vague_attribution",
            "value": f.vague_attribution_count,
            "score_impact": round(-penalty, 3),
            "triggered": penalty > 0,
            "reason": f"{f.vague_attribution_count} vague attribution(s) found (e.g. 'sources say', 'experts claim' without naming them)."
                if penalty > 0 else "No vague attribution patterns detected. Sources are either named or absent.",
            "threshold": ">0",
        })

        # 6. Pilność (-0.04 max)
        penalty = 0.0
        if f.urgency_words_count > 2:
            penalty = min(0.04, (f.urgency_words_count - 2) * 0.02)
            score -= penalty
        decisions.append({
            "feature": "urgency_language",
            "value": f.urgency_words_count,
            "score_impact": round(-penalty, 3),
            "triggered": penalty > 0,
            "reason": f"{f.urgency_words_count} urgency words/phrases found (e.g. 'share before', 'act now', 'must see'). These pressure readers to act rather than inform."
                if penalty > 0 else f"Urgency word count ({f.urgency_words_count}) is within normal range (<=2).",
            "threshold": ">2",
        })

        # 7. Superlatywy (-0.05 max, ratio-based)
        penalty = 0.0
        superlative_ratio = f.superlatives_count / max(1, f.word_count)
        if superlative_ratio > 0.02:
            penalty = min(0.05, (superlative_ratio - 0.02) * 2.5)
            score -= penalty
        decisions.append({
            "feature": "superlatives_absolutes",
            "value": {"count": f.superlatives_count, "ratio": round(superlative_ratio, 4)},
            "score_impact": round(-penalty, 3),
            "triggered": penalty > 0,
            "reason": f"{f.superlatives_count} superlatives/absolutes ({superlative_ratio:.1%} of words, detected via lexicon + POS tags JJS/RBS). Fake news overgeneralizes with 'always', 'everybody', 'worst'."
                if penalty > 0 else f"{f.superlatives_count} superlative(s) ({superlative_ratio:.1%} of words) — within normal range (<=2%).",
            "threshold": ">2% of words",
        })

        # 8. Pierwsza osoba (informational only — quotes inflate count)
        decisions.append({
            "feature": "first_person_pronouns",
            "value": round(f.first_person_ratio, 4),
            "score_impact": 0,
            "triggered": False,
            "reason": f"First-person ratio: {f.first_person_ratio:.1%} (informational only, not scored — direct quotes inflate count).",
            "threshold": "n/a",
        })

        # 9. Pytanie w tytule (-0.03)
        penalty = 0.0
        is_question = bool(title and title.strip().endswith('?'))
        if is_question:
            penalty = 0.03
            score -= penalty
        decisions.append({
            "feature": "question_headline",
            "value": is_question,
            "score_impact": round(-penalty, 3),
            "triggered": is_question,
            "reason": "Headline is a question. Per Betteridge's Law, question headlines often imply misleading answers."
                if is_question else "Headline is not a question.",
            "threshold": "headline ends with '?'",
        })

        # 10. Język konspiracyjny (-0.10 max)
        penalty = 0.0
        if f.conspiracy_phrases_count > 0:
            penalty = min(0.10, f.conspiracy_phrases_count * 0.04)
            score -= penalty
        decisions.append({
            "feature": "conspiracy_language",
            "value": {"count": f.conspiracy_phrases_count, "phrases": f.conspiracy_phrases_found},
            "score_impact": round(-penalty, 3),
            "triggered": penalty > 0,
            "reason": f"{f.conspiracy_phrases_count} conspiracy/propaganda phrase(s): {', '.join(f.conspiracy_phrases_found[:5])}. These are strongly associated with misinformation."
                if penalty > 0 else "No conspiracy or propaganda language detected.",
            "threshold": ">0",
        })

        # 11. Negation density (informational only — legitimate news uses negation heavily)
        decisions.append({
            "feature": "negation_density",
            "value": round(f.negation_ratio, 4),
            "score_impact": 0,
            "triggered": False,
            "reason": f"Negation ratio: {f.negation_ratio:.1%} (informational only, not scored — news regularly uses negation).",
            "threshold": "n/a",
        })

        # 12. Niska MATTR (-0.04 max)
        penalty = 0.0
        if f.word_count > 100 and f.lexical_diversity < 0.45:
            penalty = min(0.04, (0.45 - f.lexical_diversity) * 0.3)
            score -= penalty
        decisions.append({
            "feature": "lexical_diversity_mattr",
            "value": round(f.mattr, 3),
            "score_impact": round(-penalty, 3),
            "triggered": penalty > 0,
            "reason": f"MATTR (Moving Average TTR) is {f.mattr:.3f}. Low vocabulary diversity indicates repetitive or low-quality content."
                if penalty > 0 else f"MATTR is {f.mattr:.3f} — vocabulary diversity is adequate (>=0.45).",
            "threshold": "<0.45 (for texts >100 words)",
        })

        # 13. Title-body disconnect (-0.05)
        penalty = 0.0
        if f.title_body_coherence < 0.05 and title:
            penalty = 0.05
            score -= penalty
        decisions.append({
            "feature": "title_body_coherence",
            "value": f.title_body_coherence,
            "score_impact": round(-penalty, 3),
            "triggered": penalty > 0,
            "reason": f"TF-IDF cosine similarity {f.title_body_coherence:.2f} — title is disconnected from body content, possible clickbait."
                if penalty > 0 else f"TF-IDF cosine similarity {f.title_body_coherence:.2f} — title and body are coherent (>=0.05).",
            "threshold": "<0.05 TF-IDF similarity",
        })

        # 14. VADER sentiment (-0.05 max)
        # Only penalize extreme negativity with no factual grounding
        penalty = 0.0
        is_emotional_rant = (f.vader_compound < -0.7
                            and f.named_entity_density < 0.02
                            and f.quote_count == 0)
        if is_emotional_rant:
            penalty = min(0.05, abs(f.vader_compound + 0.7) * 0.17)
            score -= penalty
        decisions.append({
            "feature": "vader_sentiment",
            "value": {"compound": round(f.vader_compound, 3), "positive": round(f.vader_positive, 3), "negative": round(f.vader_negative, 3), "neutral": round(f.vader_neutral, 3)},
            "score_impact": round(-penalty, 3),
            "triggered": penalty > 0,
            "reason": f"VADER compound is {f.vader_compound:.3f} with no named entities or quotes — appears to be an emotional rant rather than factual reporting."
                if penalty > 0 else f"VADER compound is {f.vader_compound:.3f}. Negative sentiment alone is not penalized — news about war, crime, or disasters is naturally negative.",
            "threshold": "compound < -0.7 AND no entities AND no quotes",
        })


        # === BONUSY (podnoszą score) — total budget: ~0.78 ===

        # 17. Wskaźniki wiarygodności (+0.15 max)
        bonus = 0.0
        if f.credibility_indicators_count > 0:
            bonus = min(0.15, f.credibility_indicators_count * 0.04)
            score += bonus
        decisions.append({
            "feature": "credibility_indicators",
            "value": f.credibility_indicators_count,
            "score_impact": round(bonus, 3),
            "triggered": bonus > 0,
            "reason": f"Found {f.credibility_indicators_count} credibility indicator(s): named sources, institutional references, peer-review mentions, press releases."
                if bonus > 0 else "No credibility indicators found (no named experts, institutions, or peer-review references).",
            "threshold": ">0",
        })

        # 18. Cytaty (+0.12 max)
        bonus = 0.0
        if f.quote_count > 0:
            bonus = min(0.12, f.quote_count * 0.03)
            score += bonus
        decisions.append({
            "feature": "direct_quotes",
            "value": f.quote_count,
            "score_impact": round(bonus, 3),
            "triggered": bonus > 0,
            "reason": f"Found {f.quote_count} direct quote(s). Quoting sources directly is a hallmark of reliable journalism."
                if bonus > 0 else "No direct quotes found in the text.",
            "threshold": ">0",
        })

        # 19. Hedging (+0.10 max)
        bonus = 0.0
        if f.hedging_ratio > 0.005:
            bonus = min(0.10, f.hedging_ratio * 10)
            score += bonus
        decisions.append({
            "feature": "hedging_language",
            "value": {"ratio": round(f.hedging_ratio, 4), "count": f.hedging_words_count},
            "score_impact": round(bonus, 3),
            "triggered": bonus > 0,
            "reason": f"Hedging ratio is {f.hedging_ratio:.2%} ({f.hedging_words_count} words like 'may', 'suggests', 'approximately'). Responsible journalism qualifies uncertain claims."
                if bonus > 0 else f"Hedging ratio ({f.hedging_ratio:.2%}) is very low — text makes definitive claims without qualification.",
            "threshold": ">0.5%",
        })

        # 20. Numerical precision (+0.10 max)
        bonus = 0.0
        if f.numerical_precision_count > 0:
            bonus = min(0.10, f.numerical_precision_count * 0.025)
            score += bonus
        decisions.append({
            "feature": "numerical_precision",
            "value": f.numerical_precision_count,
            "score_impact": round(bonus, 3),
            "triggered": bonus > 0,
            "reason": f"Found {f.numerical_precision_count} precise numerical reference(s) (percentages, dollar amounts, specific years, decimal figures). Data-backed claims indicate factual reporting."
                if bonus > 0 else "No precise numerical references found (no percentages, specific figures, or dates).",
            "threshold": ">0",
        })

        # 21. Długi tekst z normalną strukturą (+0.08)
        bonus = 0.0
        well_structured = f.word_count > 500 and 15 < f.avg_sentence_length < 30
        if well_structured:
            bonus = 0.08
            score += bonus
        decisions.append({
            "feature": "text_structure",
            "value": {"word_count": f.word_count, "avg_sentence_length": round(f.avg_sentence_length, 1)},
            "score_impact": round(bonus, 3),
            "triggered": bonus > 0,
            "reason": f"Well-structured long text ({f.word_count} words, avg sentence length {f.avg_sentence_length:.1f}). Reliable articles tend to be longer with moderate sentence complexity."
                if bonus > 0 else f"Text has {f.word_count} words, avg sentence length {f.avg_sentence_length:.1f}. Does not meet well-structured criteria (>500 words, 15-30 avg sentence length).",
            "threshold": ">500 words, 15-30 avg sentence length",
        })

        # 22. Named entity density (+0.12 max)
        bonus = 0.0
        if f.named_entity_density > 0.02:
            bonus = min(0.12, (f.named_entity_density - 0.02) * 2)
            score += bonus
        ent_summary = ", ".join(f"{k}: {v}" for k, v in f.named_entity_types.items()) if f.named_entity_types else "none"
        decisions.append({
            "feature": "named_entity_density",
            "value": {"density": round(f.named_entity_density, 4), "types": f.named_entity_types},
            "score_impact": round(bonus, 3),
            "triggered": bonus > 0,
            "reason": f"Entity density is {f.named_entity_density:.2%} ({ent_summary}). Naming specific people, organizations, and places indicates factual grounding. Detected via spaCy NER."
                if bonus > 0 else f"Entity density is {f.named_entity_density:.2%} ({ent_summary}). Below the threshold for a credibility bonus (>2%).",
            "threshold": ">2%",
        })

        # 23. Umiarkowana czytelność (+0.08)
        bonus = 0.0
        moderate_readability = 40 <= f.flesch_reading_ease <= 70
        if moderate_readability:
            bonus = 0.08
            score += bonus
        interp = self._interpret_flesch(f.flesch_reading_ease)
        decisions.append({
            "feature": "readability",
            "value": {"flesch_reading_ease": round(f.flesch_reading_ease, 1), "flesch_kincaid_grade": round(f.flesch_kincaid_grade, 1), "smog_index": round(f.smog_index, 1), "coleman_liau_index": round(f.coleman_liau_index, 1), "interpretation": interp},
            "score_impact": round(bonus, 3),
            "triggered": bonus > 0,
            "reason": f"Flesch Reading Ease is {f.flesch_reading_ease:.1f} ({interp}). College-level complexity (40-70) is typical of quality journalism. Computed via textstat."
                if bonus > 0 else f"Flesch Reading Ease is {f.flesch_reading_ease:.1f} ({interp}). Outside the ideal 40-70 range for quality journalism.",
            "threshold": "40-70 Flesch Reading Ease",
        })

        # 24. Sentiment tone (informational only — no score impact)
        decisions.append({
            "feature": "sentiment_tone",
            "value": round(f.vader_compound, 3),
            "score_impact": 0.0,
            "triggered": False,
            "reason": f"VADER compound is {f.vader_compound:.3f}. Sentiment is reported for context but not penalized — legitimate news covers both positive and negative topics.",
            "threshold": "informational only",
        })

        # Ograniczenie do [0, 1]
        score = max(0.0, min(1.0, score))

        # Oblicz pewność (confidence)
        if f.word_count < 50:
            confidence = 0.3  # Za mało tekstu
        elif len(f.flags) >= 4:
            confidence = 0.9
        elif len(f.flags) >= 2:
            confidence = 0.7
        elif len(f.flags) >= 1:
            confidence = 0.5
        else:
            confidence = 0.4

        # Wygeneruj wyjaśnienie
        explanation = self._generate_explanation(f, score)

        return AnalysisResult(
            score=round(score, 3),
            confidence=round(confidence, 2),
            features=features,
            flags=f.flags,
            explanation=explanation,
            details={
                'feature_decisions': decisions,
                'basic_metrics': {
                    'word_count': f.word_count,
                    'sentence_count': f.sentence_count,
                    'avg_word_length': round(f.avg_word_length, 2),
                    'avg_sentence_length': round(f.avg_sentence_length, 1),
                },
                'clickbait_patterns': f.clickbait_patterns_found,
                'emotional_words': f.emotional_words_found,
                'conspiracy_phrases': f.conspiracy_phrases_found,
                'lexical_diversity': round(f.lexical_diversity, 3),
                'mattr': round(f.mattr, 3),
                'hedging_ratio': round(f.hedging_ratio, 4),
                'negation_ratio': round(f.negation_ratio, 4),
                'numerical_precision': f.numerical_precision_count,
                'vader_compound': round(f.vader_compound, 3),
                'vader_positive': round(f.vader_positive, 3),
                'vader_negative': round(f.vader_negative, 3),
                'flesch_reading_ease': round(f.flesch_reading_ease, 1),
                'flesch_kincaid_grade': round(f.flesch_kincaid_grade, 1),
                'smog_index': round(f.smog_index, 1),
                'coleman_liau_index': round(f.coleman_liau_index, 1),
                'named_entity_density': round(f.named_entity_density, 4),
                'named_entity_types': f.named_entity_types,
                'opening_sentences_sentiment': f.opening_sentences_sentiment,
            }
        )

    def _generate_explanation(self, f: LinguisticFeatures, score: float) -> str:
        """Wygeneruj czytelne wyjaśnienie wyniku"""

        if score >= 0.7:
            verdict = "Text shows characteristics of reliable journalism"
        elif score >= 0.5:
            verdict = "Text has some concerning features but is not clearly unreliable"
        elif score >= 0.3:
            verdict = "Text shows multiple characteristics associated with unreliable content"
        else:
            verdict = "Text shows strong characteristics of fake news or clickbait"

        details = []

        if f.clickbait_patterns_found:
            details.append("clickbait phrases detected")
        if f.emotional_words_ratio > 0.05:
            details.append(f"highly emotional language ({f.emotional_words_ratio:.1%})")
        if f.vague_attribution_count > 2:
            details.append("vague source attribution")
        if f.credibility_indicators_count > 0:
            details.append(f"{f.credibility_indicators_count} credibility indicator(s)")
        if f.quote_count > 0:
            details.append(f"{f.quote_count} direct quote(s)")
        if f.conspiracy_phrases_count > 0:
            details.append("conspiracy/propaganda language detected")
        if f.hedging_ratio > 0.005:
            details.append("uses qualifying language")
        if f.numerical_precision_count > 2:
            details.append("cites specific data points")
        if f.vader_compound < -0.3:
            details.append(f"strongly negative tone (VADER: {f.vader_compound:.2f})")
        elif f.vader_compound > 0.3:
            details.append(f"strongly positive tone (VADER: {f.vader_compound:.2f})")
        if f.named_entity_density > 0.03:
            details.append(f"mentions specific entities ({len(f.named_entity_types)} types)")
        elif f.word_count > 200 and f.named_entity_density < 0.01:
            details.append("very few named entities mentioned")

        if details:
            return f"{verdict}: {'; '.join(details)}."
        return verdict + "."

    # =========================================================================
    # METODY POMOCNICZE
    # =========================================================================

    def analyze_headline_only(self, headline: str) -> Dict:
        """
        Szybka analiza samego tytułu (dla preview).

        Returns:
            dict z oceną tytułu
        """
        headline_lower = headline.lower()

        issues = []
        score = 1.0

        # Clickbait
        for pattern in self.clickbait_patterns:
            if pattern.search(headline_lower):
                issues.append("clickbait")
                score -= 0.2
                break

        # Wykrzykniki
        if headline.count('!') > 1:
            issues.append("multiple exclamation marks")
            score -= 0.1

        # Pytanie
        if headline.strip().endswith('?'):
            issues.append("question headline")
            score -= 0.05

        # Słowa emocjonalne (spaCy tokenization)
        doc = self.nlp(headline_lower)
        words = [token.text for token in doc if token.is_alpha]
        emotional = [w for w in words if w in self.emotional_words]
        if len(emotional) > 2:
            issues.append(f"emotional: {', '.join(emotional[:3])}")
            score -= 0.1

        # VADER
        vader_scores = self.vader.polarity_scores(headline)

        return {
            'headline': headline,
            'score': max(0.0, round(score, 2)),
            'issues': issues,
            'is_clickbait': 'clickbait' in issues,
            'vader_compound': vader_scores['compound'],
        }

    def get_readability_score(self, text: str) -> Dict:
        """Oblicz metryki czytelności używając textstat."""
        if not text or len(text.split()) < 10:
            return {
                'flesch_reading_ease': 0,
                'flesch_kincaid_grade': 0,
                'smog_index': 0,
                'coleman_liau_index': 0,
                'avg_sentence_length': 0,
                'interpretation': 'Insufficient text',
            }

        flesch = textstat.flesch_reading_ease(text)
        return {
            'flesch_reading_ease': round(flesch, 1),
            'flesch_kincaid_grade': round(textstat.flesch_kincaid_grade(text), 1),
            'smog_index': round(textstat.smog_index(text), 1),
            'coleman_liau_index': round(textstat.coleman_liau_index(text), 1),
            'avg_sentence_length': round(textstat.avg_sentence_length(text), 1),
            'interpretation': self._interpret_flesch(flesch),
        }

    def _interpret_flesch(self, score: float) -> str:
        if score >= 80:
            return "Very easy (6th grade)"
        elif score >= 60:
            return "Standard (8th-9th grade)"
        elif score >= 40:
            return "Difficult (college level)"
        else:
            return "Very difficult (professional)"
