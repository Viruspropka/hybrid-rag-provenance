"""Retrieval metrics on note-level rankings. Pure functions, no I/O.

A ranking is a list of note ids, best first, without duplicates (chunk hits are
collapsed to their note in order of first appearance before scoring).

* ``recall_at_k``  = |gold notes in the top k| / |gold notes|
* ``mrr_at_k``     = 1 / rank of the first gold note within the top k, else 0
"""

from __future__ import annotations

from typing import Iterable


def collapse_to_notes(note_ids: Iterable[str]) -> list[str]:
    """Note ids in order of first appearance (chunk ranking -> note ranking)."""
    return list(dict.fromkeys(note_ids))


def recall_at_k(ranking: list[str], gold: list[str], k: int) -> float:
    if k <= 0:
        raise ValueError("k must be positive")
    g = set(gold)
    if not g:
        raise ValueError("recall is undefined without gold notes")
    return len(g & set(ranking[:k])) / len(g)


def mrr_at_k(ranking: list[str], gold: list[str], k: int = 10) -> float:
    if k <= 0:
        raise ValueError("k must be positive")
    g = set(gold)
    for i, note in enumerate(ranking[:k], 1):
        if note in g:
            return 1.0 / i
    return 0.0


def first_gold_rank(ranking: list[str], gold: list[str]) -> int | None:
    """1-based rank of the first gold note anywhere in the ranking, or None."""
    g = set(gold)
    for i, note in enumerate(ranking, 1):
        if note in g:
            return i
    return None


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0
