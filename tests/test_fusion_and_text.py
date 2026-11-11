from dataclasses import dataclass

import pytest

from hybrid_rag.fusion import diversify, rrf_score
from hybrid_rag.text import content_terms, fts5_query, term_coverage


def test_rrf_hand_computed():
    assert rrf_score([0, 0], k=60) == pytest.approx(2 / 61)
    assert rrf_score([0, None], k=60) == pytest.approx(1 / 61)
    assert rrf_score([None, None]) == 0.0
    assert rrf_score([2, 4], k=20) == pytest.approx(1 / 23 + 1 / 25)


@dataclass
class H:
    note_id: str
    text: str


def test_diversify_caps_per_note_and_collapses_twins():
    ranked = [H("a", "x one"), H("a", "x two"), H("a", "x three"), H("b", "X  One"), H("c", "z")]
    out = diversify(ranked, 10, max_per_note=2)
    assert [(h.note_id, h.text) for h in out] == [("a", "x one"), ("a", "x two"), ("c", "z")]
    assert len(diversify(ranked, 10, max_per_note=0)) == 4
    assert len(diversify(ranked, 1)) == 1


def test_fts5_query_is_safe():
    q = fts5_query('what is "the" NEAR(x) limit* per token?')
    assert '"limit"' in q and '"token"' in q and '"the"' not in q and "*" not in q
    assert fts5_query("the a of") == ""


def test_coverage():
    assert content_terms("How do the backups work, backups?") == ["backups", "work"]
    assert term_coverage("backups retention", "Backup retention is 90 days") == 1.0
    assert term_coverage("parking policy", "Backup retention") == 0.0
