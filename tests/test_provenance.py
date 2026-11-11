import math

import pytest

from hybrid_rag.config import ProvenanceConfig
from hybrid_rag.provenance import (
    authorship_weight, freshness_factor, is_historical_query, provenance_factors, status_kept, trust_factor,
)

DAY = 86400.0
NOW = 1000 * DAY


def test_trust_formula_hand_computed():
    cfg = ProvenanceConfig()
    # 0.5 + 0.5 * min(1, 0.8 * 1.0)
    assert trust_factor(0.8, "human", cfg) == pytest.approx(0.9)
    # model authorship weight 0.95: 0.5 + 0.5 * 0.8 * 0.95
    assert trust_factor(0.8, "model:x", cfg) == pytest.approx(0.88)
    # missing reliability -> default 0.85, unknown author 0.95
    assert trust_factor(None, None, cfg) == pytest.approx(0.5 + 0.5 * 0.85 * 0.95)


def test_authorship_kinds():
    cfg = ProvenanceConfig()
    assert authorship_weight("human", cfg) == 1.0
    assert authorship_weight("tool:ocr", cfg) == 0.98
    assert authorship_weight("weird", cfg) == 0.95


def test_freshness_half_life_and_floor():
    cfg = ProvenanceConfig()
    # slow half-life 365 days: one half-life old -> 0.8 + 0.2 * 0.5
    assert freshness_factor(NOW - 365 * DAY, "slow", NOW, cfg) == pytest.approx(0.9)
    # fast half-life 45 days, 90 days old -> 0.8 + 0.2 * 0.25
    assert freshness_factor(NOW - 90 * DAY, "fast", NOW, cfg) == pytest.approx(0.85)
    # very old never drops below the floor
    assert freshness_factor(0.001, "fast", NOW, cfg) >= 0.8
    # static and unknown dates are neutral
    assert freshness_factor(NOW - 9000 * DAY, "static", NOW, cfg) == 1.0
    assert freshness_factor(None, "fast", NOW, cfg) == 1.0
    # a verified date in the future counts as age 0
    assert freshness_factor(NOW + 10 * DAY, "fast", NOW, cfg) == 1.0


def test_full_multiplier_is_the_product():
    cfg = ProvenanceConfig()
    row = {"tier": 3, "reliability": 0.5, "generated_by": "model:x", "verified": NOW - 45 * DAY,
           "volatility": "fast", "status": "superseded", "doc_type": "session"}
    f = provenance_factors(row, cfg, NOW)
    expected = 0.97 * (0.5 + 0.5 * 0.5 * 0.95) * (0.8 + 0.2 * 0.5) * 0.5 * 0.6
    assert f.multiplier == pytest.approx(expected)
    assert f.as_dict()["multiplier"] == pytest.approx(round(expected, 4))


def test_switches_and_historical():
    row = {"tier": 0, "reliability": 1.0, "generated_by": "human", "verified": NOW - 400 * DAY,
           "volatility": "fast", "status": "archived", "doc_type": "note"}
    cfg = ProvenanceConfig()
    hist = provenance_factors(row, cfg, NOW, historical=True)
    assert hist.freshness == 1.0 and hist.status == 1.0
    assert provenance_factors(row, cfg, NOW).status == 0.3
    cfg.enabled = False
    assert provenance_factors(row, cfg, NOW).multiplier == 1.0
    cfg = ProvenanceConfig(trust=False, freshness=False, tier=False, doc_type=False)
    assert provenance_factors(row, cfg, NOW).multiplier == pytest.approx(0.3)


def test_unknown_status_and_type_are_neutral():
    f = provenance_factors({"status": "sent", "doc_type": "whatever", "reliability": 1.0, "generated_by": "human"},
                           ProvenanceConfig(), NOW)
    assert f.status == 1.0 and f.doc_type == 1.0 and f.tier == 1.0 and math.isclose(f.trust, 1.0)


def test_status_filter():
    cfg = ProvenanceConfig()
    assert status_kept("active", cfg, False)
    assert not status_kept("Superseded", cfg, False)
    assert status_kept("superseded", cfg, True)
    cfg.status_filter = False
    assert status_kept("archived", cfg, False)


def test_historical_query_heuristic():
    pats = [r"\b(history|previously|used to)\b"]
    assert is_historical_query("What did we previously use?", pats)
    assert not is_historical_query("What do we use?", pats)
