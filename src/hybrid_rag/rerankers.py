"""Optional rerankers. All return one float per (query, passage), higher = better.

* ``overlap``: lexical term coverage, a dependency-free stand-in for tests.
* ``cross-encoder``: sentence-transformers ``CrossEncoder`` (``BAAI/bge-reranker-v2-m3``
  by default), sigmoid scores in 0..1. Optional ``models`` extra.
* ``flashrank``: ONNX CPU rerankers via the FlashRank package. Optional extra.
"""

from __future__ import annotations

import logging
from typing import Protocol

from .config import RerankerConfig
from .text import term_coverage

log = logging.getLogger(__name__)


class Reranker(Protocol):
    name: str

    def score(self, query: str, passages: list[str]) -> list[float]: ...


class OverlapReranker:
    name = "overlap"

    def score(self, query: str, passages: list[str]) -> list[float]:
        return [term_coverage(query, p) for p in passages]


class CrossEncoderReranker:
    def __init__(self, model: str) -> None:
        import torch
        from sentence_transformers import CrossEncoder

        log.info("loading cross-encoder %s", model)
        self._model = CrossEncoder(model, max_length=1024, activation_fn=torch.nn.Sigmoid())
        self.name = f"ce:{model}"

    def score(self, query: str, passages: list[str]) -> list[float]:
        scores = self._model.predict([(query, p) for p in passages])
        return [float(s) for s in (scores.tolist() if hasattr(scores, "tolist") else scores)]


class FlashRankReranker:
    def __init__(self, model: str, cache_dir: str = "") -> None:
        from flashrank import Ranker

        kwargs = {"model_name": model, "max_length": 256}
        if cache_dir:
            kwargs["cache_dir"] = cache_dir
        self._ranker = Ranker(**kwargs)
        self.name = f"flashrank:{model}"

    def score(self, query: str, passages: list[str]) -> list[float]:
        from flashrank import RerankRequest

        ranked = self._ranker.rerank(RerankRequest(
            query=query, passages=[{"id": i, "text": p} for i, p in enumerate(passages)]))
        by_id = {int(r["id"]): float(r["score"]) for r in ranked}
        return [by_id.get(i, 0.0) for i in range(len(passages))]


def make_reranker(cfg: RerankerConfig) -> Reranker | None:
    backend = cfg.backend.lower()
    if backend in ("", "none", "off"):
        return None
    if backend == "overlap":
        return OverlapReranker()
    if backend == "cross-encoder":
        return CrossEncoderReranker(cfg.model)
    if backend == "flashrank":
        return FlashRankReranker(cfg.flashrank_model, cfg.flashrank_cache)
    raise ValueError(f"unknown reranker backend '{cfg.backend}'")
