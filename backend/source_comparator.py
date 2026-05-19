import warnings
import xml.etree.ElementTree as ET
from collections import Counter
from dataclasses import dataclass, field
from email.utils import parsedate_to_datetime
from typing import List
from urllib.parse import urlparse

import requests
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

_STOPWORDS = {
    "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "by", "from", "is", "are", "was", "were", "be", "been",
    "has", "have", "had", "that", "this", "it", "he", "she", "they", "we",
    "his", "her", "their", "its", "as", "said", "says", "according",
}

GNEWS_RSS_URL = "https://news.google.com/rss/search"


@dataclass
class MatchedSource:
    title: str
    url: str
    source: str
    published_at: str
    similarity: float


@dataclass
class CrossSourceResult:
    query: str
    sources_found: int
    verdict: str  # FOUND / NO_DATA
    matched_sources: List[MatchedSource] = field(default_factory=list)


class SourceComparator:
    """
    Uses Google News RSS to find articles covering the same story.
    No API key required. Results are re-ranked by TF-IDF similarity
    between the original title and each candidate title.
    """

    def __init__(self, nlp=None):
        self.nlp = nlp

    def _extract_key_phrases(self, title: str) -> list[str]:
        stop = {
            "a", "an", "the", "and", "or", "but", "in", "on", "at", "to",
            "for", "of", "with", "by", "from", "after", "before", "during",
            "is", "are", "was", "were", "be", "been", "has", "have", "had",
            "that", "this", "it", "he", "she", "they", "we", "his", "her",
        }
        words = [w.strip("\"'.,;:!?()") for w in title.split()]
        words = [w for w in words if w]

        phrases, run = [], []
        for w in words:
            if w.lower() in stop:
                if len(run) >= 2:
                    phrases.append(" ".join(run))
                run = []
            else:
                run.append(w)
        if len(run) >= 2:
            phrases.append(" ".join(run))

        phrases.sort(key=len, reverse=True)
        return phrases[:3]

    def _build_query(self, text: str, title: str = "") -> str:
        if title and len(title.strip()) > 10:
            phrases = self._extract_key_phrases(title)
            if phrases:
                # AND the two longest phrases only; third is optional context
                # Short titles often only yield 1-2 phrases — don't over-constrain
                must = phrases[:2]
                query = " AND ".join(f'"{p}"' for p in must)
                if len(phrases) > 2:
                    query += f' "{phrases[2]}"'
                return query
            return title.strip()

        keywords = []
        if self.nlp:
            doc = self.nlp(text[:5000])
            entities = [
                ent.text.strip()
                for ent in doc.ents
                if ent.label_ in ("PERSON", "ORG", "GPE", "EVENT", "NORP", "FAC", "LOC")
                and len(ent.text.strip()) > 2
            ]
            seen = set()
            for e in entities:
                if e.lower() not in seen:
                    seen.add(e.lower())
                    keywords.append(e)

        if not keywords:
            words = [
                w.lower() for w in text.split()
                if w.isalpha() and len(w) > 4 and w.lower() not in _STOPWORDS
            ]
            keywords = [w for w, _ in Counter(words).most_common(5)]

        terms = [f'"{kw}"' if ' ' in kw else kw for kw in keywords[:5]]
        return " OR ".join(terms)

    def _fetch_articles(self, query: str) -> list:
        try:
            resp = requests.get(
                GNEWS_RSS_URL,
                params={"q": query, "hl": "en-US", "gl": "US", "ceid": "US:en"},
                headers={"User-Agent": "Mozilla/5.0"},
                timeout=10,
            )
            resp.raise_for_status()
            root = ET.fromstring(resp.content)
            channel = root.find("channel")
            if channel is None:
                return []
            items = channel.findall("item")
        except Exception as e:
            warnings.warn(f"SourceComparator: Google News RSS failed — {e} (query: '{query}')")
            return []

        articles = []
        for item in items:
            source_el = item.find("source")
            title_el = item.find("title")
            link_el = item.find("link")
            pubdate_el = item.find("pubDate")

            source_name = source_el.text if source_el is not None else ""
            title = title_el.text or "" if title_el is not None else ""

            if source_name and title.endswith(f" - {source_name}"):
                title = title[: -(len(source_name) + 3)]

            link = link_el.text or "" if link_el is not None else ""

            try:
                pub_dt = parsedate_to_datetime(pubdate_el.text) if pubdate_el is not None else None
                published_at = pub_dt.strftime("%Y-%m-%dT%H:%M:%SZ") if pub_dt else ""
            except Exception:
                published_at = ""

            source_url = source_el.get("url", "") if source_el is not None else ""
            domain = source_url.split("//")[-1].split("/")[0].removeprefix("www.")

            if title and link:
                articles.append({
                    "title": title,
                    "url": link,
                    "domain": domain,
                    "source_name": source_name,
                    "published_at": published_at,
                })

        return articles

    def _rank_by_similarity(self, article_title: str, articles: list) -> List[float]:
        titles = [a["title"] for a in articles]
        try:
            tfidf = TfidfVectorizer(stop_words="english", max_features=2000)
            vectors = tfidf.fit_transform([article_title] + titles)
            sims = cosine_similarity(vectors[0:1], vectors[1:])[0]
            return sims.tolist()
        except ValueError:
            return [0.0] * len(titles)

    def compare(self, text: str, title: str = "", url: str = "") -> CrossSourceResult:
        query = self._build_query(text, title)
        articles = self._fetch_articles(query)

        if not articles:
            return CrossSourceResult(
                query=query,
                sources_found=0,
                verdict="NO_DATA",
            )

        similarities = self._rank_by_similarity(title or text, articles)

        # Pair each MatchedSource with its real domain (RSS urls are google redirects)
        candidates = sorted(
            [
                (
                    MatchedSource(
                        title=a["title"],
                        url=a["url"],
                        source=a["source_name"] or a["domain"],
                        published_at=a["published_at"],
                        similarity=round(float(sim), 3),
                    ),
                    a["domain"],
                )
                for a, sim in zip(articles, similarities)
            ],
            key=lambda x: x[0].similarity,
            reverse=True,
        )

        # Keep only the highest-similarity article per domain, exclude source article's domain
        article_domain = urlparse(url).netloc.removeprefix("www.")
        seen_domains: set[str] = set()
        deduped: list[MatchedSource] = []
        for s, domain in candidates:
            if domain not in seen_domains and domain != article_domain:
                seen_domains.add(domain)
                deduped.append(s)

        top = deduped[:5]
        return CrossSourceResult(
            query=query,
            sources_found=len(top),
            verdict="FOUND" if top else "NO_DATA",
            matched_sources=top,
        )
