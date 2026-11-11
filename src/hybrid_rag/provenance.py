"""Provenance ranking: frontmatter turns into bounded score multipliers.

The layer runs AFTER relevance scoring (RRF or reranker). It re-orders an
already relevant shortlist and never pulls an irrelevant chunk in. Every factor
is bounded, so a stale or low-trust note is demoted, never erased::

    final = relevance x M
    M     = tier x trust x freshness x status x doc_type

    tier      = tier_multipliers[tier]                      (missing tier -> 1.0)
    trust     = trust_floor + (1 - trust_floor) * min(1, reliability * authorship)
    freshness = freshness_floor + (1 - freshness_floor) * 0.5 ** (age_days / half_life[volatility])
                (1.0 when volatility is static, the verified date is unknown,
                 or the query is historical)
    status    = status_multipliers[status]                  (unknown word -> 1.0;
                1.0 when the query is historical: then the old note IS the answer)
    doc_type  = type_multipliers[type]                      (unknown type -> 1.0)

``authorship`` comes from ``generated_by``: the part before ``:`` is looked up in
``authorship_weights`` (``human`` 1.0, ``tool:<ocr/asr>`` 0.98, ``model:<name>``
0.95, unknown 0.95 by default). Every factor has its own switch in
``[provenance]`` so each can be measured on its own.

Separately, the status FILTER hides ``hidden_statuses`` (superseded, archived)
unless the query looks historical.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass

from .config import ProvenanceConfig

_SECONDS_PER_DAY = 86400.0


@dataclass(frozen=True)
class ProvenanceFactors:
    tier: float = 1.0
    trust: float = 1.0
    freshness: float = 1.0
    status: float = 1.0
    doc_type: float = 1.0

    @property
    def multiplier(self) -> float:
        return self.tier * self.trust * self.freshness * self.status * self.doc_type

    def as_dict(self) -> dict[str, float]:
        d = {k: round(v, 4) for k, v in asdict(self).items()}
        d["multiplier"] = round(self.multiplier, 4)
        return d


NEUTRAL = ProvenanceFactors()


def authorship_weight(generated_by: str | None, cfg: ProvenanceConfig) -> float:
    kind = (generated_by or "unknown").lower().split(":", 1)[0].strip() or "unknown"
    return cfg.authorship_weights.get(kind, cfg.authorship_weights.get("unknown", 0.95))


def freshness_factor(verified: float | None, volatility: str | None, now: float,
                     cfg: ProvenanceConfig) -> float:
    vol = (volatility or cfg.default_volatility).lower()
    if vol == "static" or not verified:
        return 1.0
    half_life = cfg.half_life_days.get(vol, cfg.half_life_days.get(cfg.default_volatility, 365.0))
    age_days = max(0.0, (now - float(verified)) / _SECONDS_PER_DAY)
    floor = cfg.freshness_floor
    return floor + (1.0 - floor) * 0.5 ** (age_days / half_life)


def trust_factor(reliability: float | None, generated_by: str | None, cfg: ProvenanceConfig) -> float:
    rel = cfg.default_reliability if reliability is None else float(reliability)
    floor = cfg.trust_floor
    return floor + (1.0 - floor) * min(1.0, max(0.0, rel) * authorship_weight(generated_by, cfg))


def provenance_factors(row: dict, cfg: ProvenanceConfig, now: float, *, historical: bool = False) -> ProvenanceFactors:
    """Factors for one chunk row (a dict with the ``chunks`` table columns)."""
    if not cfg.enabled:
        return NEUTRAL
    tier = 1.0
    if cfg.tier and row.get("tier") is not None:
        tier = cfg.tier_multipliers.get(str(row["tier"]), 1.0)
    trust = trust_factor(row.get("reliability"), row.get("generated_by"), cfg) if cfg.trust else 1.0
    fresh = 1.0
    if cfg.freshness and not historical:
        fresh = freshness_factor(row.get("verified"), row.get("volatility"), now, cfg)
    status = 1.0
    if cfg.status and not historical:
        status = cfg.status_multipliers.get((row.get("status") or "active").lower(), 1.0)
    doc_type = cfg.type_multipliers.get((row.get("doc_type") or "").lower(), 1.0) if cfg.doc_type else 1.0
    return ProvenanceFactors(tier=tier, trust=trust, freshness=fresh, status=status, doc_type=doc_type)


def status_kept(status: str | None, cfg: ProvenanceConfig, historical: bool) -> bool:
    """The status filter: hidden statuses only come back for historical queries."""
    if historical or not cfg.status_filter:
        return True
    return (status or "active").lower() not in {s.lower() for s in cfg.hidden_statuses}


def is_historical_query(query: str, patterns: list[str]) -> bool:
    return any(re.search(p, query, re.IGNORECASE) for p in patterns)
