"""Small retrieval evaluation: recall@k and MRR on labelled queries.

Queries are JSONL lines ``{"id", "query", "gold": [note ids], "tag"}``.
Ranking is evaluated at NOTE level: chunk hits are collapsed to their note in
order of first appearance.

* ``recall@k`` = |gold notes in the top k| / |gold notes|, averaged over queries
* ``MRR``      = 1 / rank of the first gold note (0 when none in the top ``depth``)
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from .config import Config
from .embedders import Embedder, make_embedder
from .retriever import Retriever


@dataclass
class Variant:
    name: str
    mode: str = "hybrid"
    provenance: bool = False          # multipliers on/off
    status_filter: bool = False


DEFAULT_VARIANTS = [
    Variant("FTS5 only (BM25)", mode="sparse"),
    Variant("Dense only", mode="dense"),
    Variant("Hybrid RRF, no provenance", mode="hybrid"),
    Variant("Hybrid RRF + status filter", mode="hybrid", status_filter=True),
    Variant("Hybrid RRF + full provenance", mode="hybrid", provenance=True, status_filter=True),
]


def load_queries(path: str | Path) -> list[dict]:
    out = []
    for n, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        item = json.loads(line)
        if not item.get("query") or not item.get("gold"):
            raise ValueError(f"{path}:{n}: each query needs 'query' and a non-empty 'gold' list")
        out.append(item)
    return out


def note_ranking(note_ids: list[str]) -> list[str]:
    return list(dict.fromkeys(note_ids))


def recall_at_k(ranking: list[str], gold: list[str], k: int) -> float:
    g = set(gold)
    return len(g & set(ranking[:k])) / len(g) if g else 0.0


def reciprocal_rank(ranking: list[str], gold: list[str]) -> float:
    g = set(gold)
    for i, note in enumerate(ranking, 1):
        if note in g:
            return 1.0 / i
    return 0.0


@dataclass
class VariantResult:
    name: str
    metrics: dict[str, float]
    per_query: list[dict] = field(default_factory=list)


def run_variant(cfg: Config, variant: Variant, queries: list[dict], *, embedder: Embedder,
                ks: tuple[int, ...] = (1, 3, 5), depth: int = 10, fastpath: bool = False) -> VariantResult:
    vcfg = cfg.copy()
    vcfg.provenance.enabled = variant.provenance
    vcfg.provenance.status_filter = variant.status_filter
    if not fastpath:
        vcfg.fastpath.exact_key_pattern = ""
    retriever = Retriever(vcfg, embedder=embedder, memory_probe=lambda: None)
    per_query = []
    sums = {f"recall@{k}": 0.0 for k in ks} | {"MRR": 0.0}
    for q in queries:
        res = retriever.retrieve(q["query"], depth, mode=variant.mode)
        ranking = note_ranking([h.note_id for h in res.hits])
        row = {"id": q.get("id"), "tag": q.get("tag", ""), "top": ranking[:5], "gold": q["gold"],
               "rr": reciprocal_rank(ranking, q["gold"]), "mode": res.mode}
        for k in ks:
            r = recall_at_k(ranking, q["gold"], k)
            row[f"recall@{k}"] = r
            sums[f"recall@{k}"] += r
        sums["MRR"] += row["rr"]
        per_query.append(row)
    n = len(queries) or 1
    return VariantResult(variant.name, {key: value / n for key, value in sums.items()}, per_query)


def run_eval(cfg: Config, queries: list[dict], variants: list[Variant] | None = None, *,
             embedder: Embedder | None = None, ks: tuple[int, ...] = (1, 3, 5)) -> list[VariantResult]:
    emb = embedder or make_embedder(cfg.embedder)
    return [run_variant(cfg, v, queries, embedder=emb, ks=ks) for v in (variants or DEFAULT_VARIANTS)]


def markdown_table(results: list[VariantResult]) -> str:
    if not results:
        return ""
    keys = list(results[0].metrics)
    lines = ["| Configuration | " + " | ".join(keys) + " |",
             "|---|" + "---:|" * len(keys)]
    for r in results:
        lines.append(f"| {r.name} | " + " | ".join(f"{r.metrics[k]:.3f}" for k in keys) + " |")
    return "\n".join(lines)
