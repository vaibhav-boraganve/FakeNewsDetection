"""Text / newspaper preprocessing with NLTK.

Uses only NLTK components that work without downloaded corpora (regex tokenizer,
Porter stemmer). The NLTK stopword list is used when available, otherwise a
built-in English list is used, so the pipeline also runs fully offline.
"""
from __future__ import annotations

import html
import re
from functools import lru_cache

from nltk.stem import PorterStemmer
from nltk.tokenize import RegexpTokenizer, sent_tokenize

_URL_RE = re.compile(r"https?://\S+|www\.\S+")
_HTML_RE = re.compile(r"<[^>]+>")
_NON_ALPHA_RE = re.compile(r"[^a-z\s']")
_WS_RE = re.compile(r"\s+")
_SENT_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")

_tokenizer = RegexpTokenizer(r"[a-z]+(?:'[a-z]+)?")
_stemmer = PorterStemmer()

_FALLBACK_STOPWORDS = set(
    """a about above after again against all am an and any are as at be because been before being below
    between both but by can did do does doing down during each few for from further had has have having he
    her here hers herself him himself his how i if in into is it its itself just me more most my myself no
    nor not now of off on once only or other our ours ourselves out over own same she should so some such
    than that the their theirs them themselves then there these they this those through to too under until
    up very was we were what when where which while who whom why will with you your yours yourself
    yourselves would could also s t""".split()
)

SENSATIONAL_TERMS = {
    "shocking", "unbelievable", "miracle", "secret", "exposed", "banned", "hoax", "conspiracy",
    "outrage", "bombshell", "urgent", "share", "viral", "insane", "destroyed", "slams", "coverup",
    "mainstream", "sheeple", "cure", "cures", "instantly", "deleted", "censored", "elites",
    "globalist", "plandemic", "agenda", "furious", "stunned", "silenced",
}
CLICKBAIT_PHRASES = (
    "you won't believe", "what happened next", "doctors hate", "share before", "before it's deleted",
    "they don't want you to know", "mainstream media won't", "wake up", "100% proof", "this one trick",
    "share this", "going viral", "the truth about", "exposed", "must watch", "must read",
)
ATTRIBUTION_PHRASES = (
    "according to", "said in a statement", "told reporters", "spokesperson", "officials said",
    "reported by", "data from", "the ministry", "the department", "percent", "per cent", "survey",
)


@lru_cache(maxsize=1)
def get_stopwords() -> frozenset[str]:
    try:
        from nltk.corpus import stopwords

        return frozenset(stopwords.words("english"))
    except LookupError:
        return frozenset(_FALLBACK_STOPWORDS)


def clean_text(text: str) -> str:
    """Normalise raw article text: unescape HTML, strip tags/URLs, lowercase, keep letters."""
    text = html.unescape(text or "")
    text = _HTML_RE.sub(" ", text)
    text = _URL_RE.sub(" ", text)
    text = text.lower()
    text = _NON_ALPHA_RE.sub(" ", text)
    return _WS_RE.sub(" ", text).strip()


def tokenize(text: str, remove_stopwords: bool = True, stem: bool = True) -> list[str]:
    tokens = _tokenizer.tokenize(clean_text(text))
    if remove_stopwords:
        sw = get_stopwords()
        tokens = [t for t in tokens if t not in sw and len(t) > 1]
    if stem:
        tokens = [_stemmer.stem(t) for t in tokens]
    return tokens


def preprocess_for_tfidf(text: str) -> str:
    """Analyzer input for the TF-IDF model: cleaned, stopword-free, stemmed tokens."""
    return " ".join(tokenize(text))


def split_sentences(text: str) -> list[str]:
    try:
        return [s for s in sent_tokenize(text) if s.strip()]
    except LookupError:
        return [s for s in _SENT_SPLIT_RE.split(text.strip()) if s.strip()]


def linguistic_features(text: str) -> dict[str, float]:
    """Stylometric cues commonly associated with fabricated or sensational news."""
    raw = text or ""
    lower = raw.lower()
    words = re.findall(r"[A-Za-z']+", raw)
    n_words = max(len(words), 1)
    sentences = split_sentences(raw) or [raw]
    tokens = _tokenizer.tokenize(clean_text(raw))

    caps_words = sum(1 for w in words if len(w) > 2 and w.isupper())
    return {
        "word_count": float(len(words)),
        "avg_sentence_length": len(words) / max(len(sentences), 1),
        "exclamation_ratio": raw.count("!") / max(len(sentences), 1),
        "question_ratio": raw.count("?") / max(len(sentences), 1),
        "all_caps_ratio": caps_words / n_words,
        "sensational_ratio": sum(1 for t in tokens if t in SENSATIONAL_TERMS) / n_words,
        "clickbait_hits": float(sum(1 for p in CLICKBAIT_PHRASES if p in lower)),
        "attribution_hits": float(sum(1 for p in ATTRIBUTION_PHRASES if p in lower)),
        "lexical_diversity": len(set(tokens)) / max(len(tokens), 1),
        "digit_ratio": sum(c.isdigit() for c in raw) / max(len(raw), 1),
        "quote_count": float(raw.count('"') // 2 + raw.count("\u201c")),
    }
