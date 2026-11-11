"""Reciprocal-rank fusion and result diversification."""

from __future__ import annotations

from typing import Protocol, TypeVar


def rrf_score(ranks: list[int | None], k: int = 60) -> float:
    """Reciprocal-rank fusion over 0-based ranks; ``None`` = absent from that list.

    ``score = sum(1 / (k + rank + 1))`` (Cormack, Clarke and Buettcher, 2009).
    A larger ``k`` flattens the merge.
    """
    return sum(1.0 / (k + r + 1) for r in ranks if r is not None)


class _HasNote(Protocol):
    note_id: str
    text: str


H = TypeVar("H", bound=_HasNote)


def diversify(ranked: list[H], k: int, max_per_note: int = 2) -> list[H]:
    """First ``k`` of ``ranked`` with identical texts collapsed and a per-note cap.

    Keeps one note from filling every slot with its own chunks; the freed slots
    are refilled by the next candidates. ``max_per_note`` 0 turns the cap off.
    """
    out: list[H] = []
    per_note: dict[str, int] = {}
    seen_text: set[str] = set()
    for h in ranked:
        key = " ".join(h.text.lower().split())
        if key in seen_text:
            continue
        seen_text.add(key)
        if max_per_note > 0 and per_note.get(h.note_id, 0) >= max_per_note:
            continue
        per_note[h.note_id] = per_note.get(h.note_id, 0) + 1
        out.append(h)
        if len(out) >= k:
            break
    return out
