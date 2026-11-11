"""Small shared text helpers: tokenisation, stopwords, FTS5-safe query building."""

from __future__ import annotations

import re

_WORD_RE = re.compile(r"[^\W_]+", re.UNICODE)

STOPWORDS = frozenset("""
a an and are as at be been but by can could did do does for from had has have how i if in into is it
its me my no not of on or our should so than that the their them then there these they this to
us was we were what when where which who whom why will with would you your about any after
before done also just more most other some such only own same too very per much many now s t
""".split())


def tokens(text: str) -> list[str]:
    """Lowercased word tokens (unicode letters and digits)."""
    return [t.lower() for t in _WORD_RE.findall(text)]


def content_terms(text: str) -> list[str]:
    """Tokens without stopwords and one-character noise, order kept, duplicates removed."""
    seen: dict[str, None] = {}
    for t in tokens(text):
        if len(t) >= 2 and t not in STOPWORDS:
            seen.setdefault(t, None)
    return list(seen)


def fts5_query(text: str) -> str:
    """OR-query of quoted content terms, so user input can never break FTS5 syntax."""
    terms = content_terms(text)
    return " OR ".join(f'"{t}"' for t in terms)


def light_stem(term: str) -> str:
    """Very small English suffix stripper, used only for coverage checks."""
    for suffix in ("ing", "ies", "es", "ed", "s"):
        if len(term) > len(suffix) + 2 and term.endswith(suffix):
            return term[: -len(suffix)]
    return term


def term_coverage(query: str, text: str) -> float:
    """Share of the query's content terms that appear (stemmed) in ``text``; 0..1."""
    q = {light_stem(t) for t in content_terms(query)}
    if not q:
        return 0.0
    d = {light_stem(t) for t in tokens(text)}
    return len(q & d) / len(q)
