import re
from typing import List, Optional
from dataclasses import dataclass, field

import spacy

# Phrases to strip from claims before sending to API
_ATTRIBUTION_PATTERNS = [
    r'^according\s+to\s+[^,]+,\s*',
    r'^(.*,\s*)?lead\s+\w+(\s+\w+)?\s+(at|of|for)\s+[^,]+,\s*',
    r'^[^,]*\b(said|stated|claimed|announced|reported|confirmed|denied)\s+that\s+',
    r'^[^,]*\b(said|stated|claimed|announced|reported|confirmed|denied)\s*,?\s*[""\u201c]',
    r'^the\s+[A-Z][a-z]+(\s+[A-Z][a-z]+)*\s+(confirmed|announced|reported|said|stated|denied)\s+that\s+',
    r'^[A-Z][a-z]+(\s+[A-Z][a-z]+)*\s+(claimed|said|stated|announced|reported)\s+that\s+',
    r',?\s*(said|stated|claimed|announced|reported)\s+[A-Z][a-z]+.*$',
    r',?\s*according\s+to\s+.*$',
    r'\b(a|an|the)\s+new\s+(study|report|research|survey)\s+(published\s+in\s+[^,]+\s+)?(shows?|found|reveals?|suggests?|indicates?)\s+that\s+',
]

# Filler words to remove from search queries
_FILLER_WORDS = {
    'the', 'a', 'an', 'is', 'are', 'was', 'were', 'has', 'have', 'had',
    'been', 'being', 'that', 'this', 'those', 'these', 'with', 'from',
    'for', 'and', 'but', 'its', 'their', 'also', 'more', 'than',
    'about', 'into', 'over', 'such', 'will', 'would', 'could',
    'should', 'may', 'might', 'can', 'which', 'what', 'when', 'where',
    'who', 'whom', 'how', 'very', 'just', 'some', 'any', 'each',
}


@dataclass
class Claim:
    text: str
    score: float
    search_query: str = ""  # Simplified query for API search
    entities: List[str] = field(default_factory=list)
    entity_names: List[str] = field(default_factory=list)  # Raw entity text
    sentence_index: int = 0


class ClaimExtractor:
    FACT_ENTITIES = {"PERSON", "ORG", "GPE", "DATE", "MONEY", "PERCENT",
                     "QUANTITY", "CARDINAL", "EVENT", "LAW", "NORP"}

    CLAIM_PATTERNS = [
        r'\d+\s*%',
        r'\d+\s*(million|billion|thousand)',
        r'(said|stated|claimed|announced|reported)\s+that',
        r'according to',
        r'(study|research|report|survey)\s+(shows?|found|reveals?)',
        r'(confirmed|denied|admitted)',
    ]

    REJECT_PATTERNS = [
        r'^\s*$',
        r'\?\s*$',
        r'^(and|but|or|if|when)\s',
    ]

    _attribution_compiled = [re.compile(p, re.IGNORECASE) for p in _ATTRIBUTION_PATTERNS]

    def __init__(self, nlp: Optional[spacy.language.Language] = None):
        """Accept a shared spaCy model instead of loading a duplicate."""
        if nlp is not None:
            self.nlp = nlp
        else:
            try:
                self.nlp = spacy.load("en_core_web_sm")
            except OSError:
                self.nlp = None

    def extract(self, text: str, top_n: int = 5, min_score: float = 0.3) -> List[Claim]:
        if self.nlp is None:
            return self._fallback_extract(text, top_n)

        doc = self.nlp(text)
        scored_claims = []
        seen_texts = set()

        for i, sent in enumerate(doc.sents):
            sent_text = sent.text.strip()

            if len(sent_text) < 25:
                continue
            if any(re.search(p, sent_text, re.IGNORECASE) for p in self.REJECT_PATTERNS):
                continue

            normalized = re.sub(r'\s+', ' ', sent_text.lower())
            if normalized in seen_texts:
                continue
            seen_texts.add(normalized)

            score, entities, entity_names = self._score_sentence(sent)

            if score >= min_score:
                search_query = self._build_search_query(sent_text, entity_names)
                scored_claims.append(Claim(
                    text=sent_text,
                    score=score,
                    search_query=search_query,
                    entities=entities,
                    entity_names=entity_names,
                    sentence_index=i
                ))

        scored_claims.sort(key=lambda x: x.score, reverse=True)
        return scored_claims[:top_n]

    def _score_sentence(self, sent) -> tuple[float, List[str], List[str]]:
        score = 0.2
        entities = []
        entity_names = []

        for ent in sent.ents:
            if ent.label_ in self.FACT_ENTITIES:
                entities.append(f"{ent.text} ({ent.label_})")
                entity_names.append(ent.text)
                if ent.label_ == "PERSON":
                    score += 0.15
                elif ent.label_ in {"ORG", "GPE"}:
                    score += 0.12
                elif ent.label_ in {"DATE", "MONEY", "PERCENT", "QUANTITY"}:
                    score += 0.18
                else:
                    score += 0.08

        sent_text = sent.text
        for pattern in self.CLAIM_PATTERNS:
            if re.search(pattern, sent_text, re.IGNORECASE):
                score += 0.12

        numbers_count = sum(1 for token in sent if token.like_num)
        score += min(0.15, numbers_count * 0.05)

        if '"' in sent_text or '\u201c' in sent_text or '\u201e' in sent_text:
            score += 0.1

        word_count = len([t for t in sent if not t.is_punct])
        if word_count > 15:
            score += 0.05
        if word_count > 25:
            score += 0.05

        return min(1.0, score), entities, entity_names

    def _build_search_query(self, sentence: str, entity_names: List[str]) -> str:
        """
        Transform a full sentence into a short, keyword-focused search query.

        Strategy:
        1. Strip attribution phrases ("according to X", "X said that")
        2. Remove filler words
        3. Keep entities, numbers, and key nouns/verbs
        4. Cap at ~12 words
        """
        query = sentence

        # Strip attribution wrappers
        for pattern in self._attribution_compiled:
            query = pattern.sub('', query)

        query = query.strip(' ,.\u201c\u201d"\'')

        # Remove quotes
        query = re.sub(r'["\u201c\u201d\u201e\u201f]', '', query)

        # Split into words, remove filler
        words = query.split()
        keywords = []
        for w in words:
            w_lower = w.lower().strip('.,;:!?()')
            if w_lower in _FILLER_WORDS:
                continue
            keywords.append(w.strip('.,;:!?()'))

        # Ensure entities are present
        for ent in entity_names:
            ent_words = ent.split()
            if not any(ew.lower() in [k.lower() for k in keywords] for ew in ent_words):
                keywords.extend(ent_words)

        # Cap length: short queries work better with the API
        if len(keywords) > 12:
            keywords = keywords[:12]

        result = ' '.join(keywords)

        # Fallback: if stripping made it too short, use first 10 words of original
        if len(result.split()) < 3:
            result = ' '.join(sentence.split()[:10])

        return result

    def _fallback_extract(self, text: str, top_n: int) -> List[Claim]:
        sentences = re.split(r'(?<=[.!?])\s+', text)
        claims = []

        for i, sent in enumerate(sentences):
            sent = sent.strip()
            if len(sent) < 25:
                continue

            score = 0.3
            if re.search(r'\d+%', sent):
                score += 0.2
            if re.search(r'(said|claimed|according)', sent, re.IGNORECASE):
                score += 0.15
            if any(c.isupper() for c in sent[1:]):
                score += 0.1

            # Simple query: first 10 words
            search_query = ' '.join(sent.split()[:10])

            claims.append(Claim(
                text=sent,
                score=min(1.0, score),
                search_query=search_query,
                sentence_index=i,
            ))

        claims.sort(key=lambda x: x.score, reverse=True)
        return claims[:top_n]
