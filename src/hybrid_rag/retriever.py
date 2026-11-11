"""Hybrid retrieval with provenance ranking and a degrade ladder.

Pipeline for one query::

    fast paths (exact keys)      -> deterministic rows first, score 1.0
    dense  (embed + cosine top-N) \
                                   > reciprocal-rank fusion -> status filter
    sparse (FTS5 BM25 top-N)      /
    -> optional reranker (with timeout and a "flat scores" guard)
    -> relevance x provenance multiplier -> per-note diversification -> top k

Degrade ladder: when the dense side cannot run (index file missing, index built
by another embedder, embedder fails to load, encode exceeds its time budget,
memory pressure) the query is answered by keyword search alone, and the result
says so in ``mode`` and ``reason``. A reranker failure or timeout falls back to
the fused order instead. Nothing on this path blocks for minutes.
"""

from __future__ import annotations

import logging
import sqlite3
import threading
import time
from dataclasses import asdict, dataclass, field
from typing import Callable

from . import store
from .config import Config
from .embedders import Embedder, make_embedder
from .fastpath import FastPath, make_fastpaths
from .fusion import diversify, rrf_score
from .provenance import NEUTRAL, provenance_factors, status_kept, is_historical_query
from .rerankers import Reranker, make_reranker
from .text import fts5_query, term_coverage

log = logging.getLogger(__name__)

MODES = ("hybrid", "sparse", "dense")


class InferenceTimeout(RuntimeError):
    """A model stage overran its time budget."""


class DenseUnavailable(RuntimeError):
    """The dense side cannot answer; the caller degrades to keyword search."""


@dataclass
class Hit:
    chunk_id: str
    note_id: str
    file_path: str
    title: str
    section_heading: str
    text: str
    status: str
    doc_type: str
    score: float                    # final = relevance x provenance multiplier
    relevance: float                # normalised relevance before provenance
    fused: float = 0.0              # raw RRF score (0 on single-retriever paths)
    dense_rank: int | None = None   # 0-based rank in the dense list
    sparse_rank: int | None = None  # 0-based rank in the BM25 list
    rerank: float | None = None
    sources: list[str] = field(default_factory=list)
    provenance: dict[str, float] = field(default_factory=dict)
    coverage: float = 0.0           # share of query terms present in the chunk

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class RetrievalResult:
    query: str
    hits: list[Hit]
    mode: str                       # hybrid | sparse | dense | degraded | fastpath
    reason: str = ""                # why it degraded, if it did
    historical: bool = False
    low_confidence: bool = False
    timings: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["hits"] = [h.to_dict() for h in self.hits]
        return d


def run_with_timeout(fn: Callable[[], object], timeout_s: float, label: str) -> object:
    """Run ``fn`` on a daemon thread; raise ``InferenceTimeout`` past the budget.

    Python threads cannot be killed, so an overrunning worker is abandoned and
    finishes in the background; the caller degrades instead of waiting.
    """
    box: dict[str, object] = {}

    def target() -> None:
        try:
            box["result"] = fn()
        except BaseException as exc:  # noqa: BLE001 - re-raised in the caller
            box["error"] = exc

    t = threading.Thread(target=target, name=f"hrag-{label}", daemon=True)
    t.start()
    t.join(timeout_s)
    if t.is_alive():
        raise InferenceTimeout(f"{label} exceeded {timeout_s:.1f}s")
    if "error" in box:
        raise box["error"]  # type: ignore[misc]
    return box.get("result")


def memory_pressure_reason(min_available_pct: float, max_swap_pct: float) -> str | None:
    """A reason string when RAM is too tight for model inference; fails open without psutil."""
    try:
        import psutil
    except ImportError:
        return None
    try:
        vm = psutil.virtual_memory()
        avail = vm.available * 100.0 / vm.total
        if avail < min_available_pct:
            return f"available RAM {avail:.0f}% < {min_available_pct:.0f}%"
        swap = psutil.swap_memory()
        if swap.total and swap.used * 100.0 / swap.total > max_swap_pct:
            return f"swap used {swap.used * 100.0 / swap.total:.0f}% > {max_swap_pct:.0f}%"
    except (OSError, RuntimeError, AttributeError, ValueError) as exc:
        log.warning("memory probe failed (%s); continuing", exc)
    return None


class Retriever:
    """Thread-safe retriever over one index directory."""

    def __init__(self, cfg: Config, *, embedder: Embedder | None = None,
                 reranker: Reranker | None = None, fastpaths: list[FastPath] | None = None,
                 memory_probe: Callable[[], str | None] | None = None) -> None:
        self.cfg = cfg
        self._conns = store.ThreadLocalConnections(cfg.index_path)
        self._embedder = embedder
        self._embedder_error: str | None = None
        self._reranker = reranker
        self._reranker_loaded = reranker is not None
        self._dense: store.DenseIndex | None = None
        self._dense_mtime: float | None = None
        self._lock = threading.Lock()
        self.fastpaths = make_fastpaths(cfg.fastpath.exact_key_pattern) if fastpaths is None else fastpaths
        r = cfg.retrieval
        self._memory_probe = memory_probe or (
            lambda: memory_pressure_reason(r.mem_min_available_pct, r.mem_max_swap_used_pct))

    # ------------------------------------------------------------- lazy parts
    def embedder(self) -> Embedder:
        with self._lock:
            if self._embedder is None:
                if self._embedder_error:
                    raise DenseUnavailable(self._embedder_error)
                try:
                    self._embedder = make_embedder(self.cfg.embedder)
                except Exception as exc:  # noqa: BLE001 - any load failure degrades
                    self._embedder_error = f"embedder failed to load: {exc}"
                    raise DenseUnavailable(self._embedder_error) from exc
            return self._embedder

    def dense_index(self) -> store.DenseIndex:
        path = self.cfg.dense_path
        try:
            mtime = path.stat().st_mtime
        except OSError as exc:
            raise DenseUnavailable("dense index file missing") from exc
        with self._lock:
            if self._dense is None or self._dense_mtime != mtime:
                try:
                    self._dense = store.DenseIndex.load(path, self.cfg.dense_ids_path)
                except (OSError, ValueError, KeyError) as exc:
                    raise DenseUnavailable(f"dense index unreadable: {exc}") from exc
                self._dense_mtime = mtime
            return self._dense

    def reranker(self) -> Reranker | None:
        with self._lock:
            if not self._reranker_loaded:
                self._reranker_loaded = True
                try:
                    self._reranker = make_reranker(self.cfg.reranker)
                except Exception as exc:  # noqa: BLE001 - no reranker is a valid state
                    log.warning("reranker unavailable (%s); using fused order", exc)
                    self._reranker = None
            return self._reranker

    def warm(self) -> None:
        """Load models up front (the daemon calls this so the first query is fast)."""
        try:
            self.embedder()
            self.dense_index()
        except DenseUnavailable as exc:
            log.warning("dense side not available at start: %s", exc)
        self.reranker()

    def conn(self) -> sqlite3.Connection:
        return self._conns.get()

    # ---------------------------------------------------------------- helpers
    def _rows(self, chunk_ids: list[str]) -> dict[str, dict]:
        if not chunk_ids:
            return {}
        marks = ",".join("?" * len(chunk_ids))
        rows = self.conn().execute(f"SELECT * FROM chunks WHERE chunk_id IN ({marks})", chunk_ids).fetchall()
        return {r["chunk_id"]: dict(r) for r in rows}

    def _sparse_ranked(self, query: str, limit: int) -> list[tuple[str, float]]:
        q = fts5_query(query)
        if not q:
            return []
        try:
            rows = self.conn().execute(
                "SELECT chunk_id, bm25(chunks_fts, 1.0, 0.5) AS rank FROM chunks_fts "
                "WHERE chunks_fts MATCH ? ORDER BY rank, chunk_id LIMIT ?", (q, limit)).fetchall()
        except sqlite3.OperationalError as exc:
            log.warning("FTS5 query failed (%s)", exc)
            return []
        return [(r["chunk_id"], -float(r["rank"])) for r in rows]   # negate: higher = better

    def _dense_ranked(self, query: str, limit: int, timings: dict) -> list[tuple[str, float]]:
        index = self.dense_index()
        emb = self.embedder()
        if index.embedder != emb.name:
            raise DenseUnavailable(f"index built with '{index.embedder}', query embedder is '{emb.name}'; re-index")
        t = time.perf_counter()
        try:
            qvec = run_with_timeout(lambda: emb.encode([query])[0], self.cfg.retrieval.inference_timeout_s, "embed")
        except InferenceTimeout as exc:
            raise DenseUnavailable(str(exc)) from exc
        except Exception as exc:  # noqa: BLE001 - model runtime errors are diverse
            raise DenseUnavailable(f"embedding failed: {exc}") from exc
        timings["embed"] = time.perf_counter() - t
        t = time.perf_counter()
        out = index.search(qvec, limit)
        timings["dense"] = time.perf_counter() - t
        return out

    def _make_hit(self, row: dict, relevance: float, now: float, historical: bool, query: str,
                  **extra: object) -> Hit:
        factors = provenance_factors(row, self.cfg.provenance, now, historical=historical)
        text_for_cov = f"{row['title']} {row['section_heading']} {row['text']}"
        return Hit(
            chunk_id=row["chunk_id"], note_id=row["note_id"], file_path=row["file_path"], title=row["title"],
            section_heading=row["section_heading"], text=row["text"], status=row["status"],
            doc_type=row["doc_type"], score=relevance * factors.multiplier, relevance=relevance,
            provenance=factors.as_dict(), coverage=round(term_coverage(query, text_for_cov), 4),
            **extra,  # type: ignore[arg-type]
        )

    def _finish(self, query: str, hits: list[Hit], k: int, mode: str, reason: str, historical: bool,
                timings: dict, fast: list[Hit] | None = None) -> RetrievalResult:
        hits.sort(key=lambda h: (-h.score, h.chunk_id))
        ranked = diversify(hits, len(hits), self.cfg.retrieval.max_chunks_per_note)
        fast = fast or []
        seen = {h.chunk_id for h in fast}
        merged = fast + [h for h in ranked if h.chunk_id not in seen]
        final = merged[:k]
        low = not final or (not fast and final[0].coverage < self.cfg.retrieval.low_confidence_coverage)
        return RetrievalResult(query=query, hits=final, mode=mode, reason=reason, historical=historical,
                               low_confidence=low, timings={k_: round(v, 4) for k_, v in timings.items()})

    # ------------------------------------------------------------------ paths
    def _fast_hits(self, query: str, k: int) -> tuple[list[Hit], bool]:
        hits: list[Hit] = []
        covered = False
        for fp in self.fastpaths:
            try:
                rows = fp.rows(query, self.conn(), k)
            except Exception as exc:  # noqa: BLE001 - a plugin must never break retrieval
                log.warning("fast path %s failed (%s)", fp.name, exc)
                continue
            for r in rows:
                row = dict(r)
                hits.append(Hit(
                    chunk_id=row["chunk_id"], note_id=row["note_id"], file_path=row["file_path"],
                    title=row["title"], section_heading=row["section_heading"], text=row["text"],
                    status=row["status"], doc_type=row["doc_type"], score=1.0, relevance=1.0,
                    sources=[f"fastpath:{fp.name}"], provenance=NEUTRAL.as_dict(),
                    coverage=round(term_coverage(query, row["text"]), 4)))
            covered = covered or (bool(rows) and fp.covers_query(query))
        return hits[:k], covered

    def _sparse_only(self, query: str, k: int, historical: bool, now: float, timings: dict) -> list[Hit]:
        t = time.perf_counter()
        ranked = self._sparse_ranked(query, max(k * self.cfg.retrieval.sparse_overfetch, 20))
        timings["sparse"] = time.perf_counter() - t
        rows = self._rows([cid for cid, _ in ranked])
        best = max((s for _, s in ranked), default=0.0) or 1.0
        hits = []
        for rank, (cid, bm25) in enumerate(ranked):
            row = rows.get(cid)
            if row is None or not status_kept(row["status"], self.cfg.provenance, historical):
                continue
            hits.append(self._make_hit(row, max(bm25, 0.0) / best, now, historical, query,
                                       sparse_rank=rank, sources=["sparse"]))
        return hits

    def retrieve(self, query: str, k: int = 5, *, historical: bool | None = None, mode: str = "hybrid",
                 force_sparse: bool = False) -> RetrievalResult:
        """Top-``k`` hits for ``query``. ``mode`` is hybrid (default), sparse or dense (for evaluation)."""
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}")
        query = query.strip()
        timings: dict[str, float] = {}
        t_all = time.perf_counter()
        if historical is None:
            historical = is_historical_query(query, self.cfg.retrieval.historical_patterns)
        if not query:
            return RetrievalResult(query=query, hits=[], mode=mode, historical=historical, low_confidence=True)
        now = self.cfg.reference_epoch()

        fast, covered = self._fast_hits(query, k) if mode == "hybrid" else ([], False)
        if fast and covered:
            timings["total"] = time.perf_counter() - t_all
            return self._finish(query, [], k, "fastpath", "", historical, timings, fast)

        reason = ""
        if force_sparse or mode == "sparse":
            reason = "forced keyword-only" if force_sparse else ""
            hits = self._sparse_only(query, k, historical, now, timings)
            timings["total"] = time.perf_counter() - t_all
            return self._finish(query, hits, k, "degraded" if force_sparse else "sparse", reason,
                                historical, timings, fast)

        pressure = self._memory_probe()
        try:
            if pressure:
                raise DenseUnavailable(f"memory pressure: {pressure}")
            dense = self._dense_ranked(query, self.cfg.retrieval.dense_top_k, timings)
        except DenseUnavailable as exc:
            if mode == "dense":
                raise
            log.warning("dense side unavailable (%s): keyword-only", exc)
            hits = self._sparse_only(query, k, historical, now, timings)
            timings["total"] = time.perf_counter() - t_all
            return self._finish(query, hits, k, "degraded", str(exc), historical, timings, fast)

        sparse: list[tuple[str, float]] = []
        if mode == "hybrid":
            t = time.perf_counter()
            sparse = self._sparse_ranked(query, self.cfg.retrieval.sparse_top_k)
            timings["sparse"] = time.perf_counter() - t

        d_rank = {cid: i for i, (cid, _) in enumerate(dense)}
        s_rank = {cid: i for i, (cid, _) in enumerate(sparse)}
        ids = list(dict.fromkeys([c for c, _ in dense] + [c for c, _ in sparse]))
        rows = self._rows(ids)
        rrf_k = self.cfg.retrieval.rrf_k
        cands = []
        for cid in ids:
            row = rows.get(cid)
            if row is None or not status_kept(row["status"], self.cfg.provenance, historical):
                continue
            fused = rrf_score([d_rank.get(cid), s_rank.get(cid)], rrf_k)
            cands.append((fused, cid, row))
        cands.sort(key=lambda c: (-c[0], c[1]))
        if not cands:
            timings["total"] = time.perf_counter() - t_all
            return self._finish(query, [], k, mode, reason, historical, timings, fast)

        # Relevance = fused score / best fused score: the top candidate is 1.0 and a chunk
        # found by only one retriever sits near 0.5. RRF keeps ranks, not magnitudes, so
        # rrf_k decides how much room the multipliers get: with the paper's k=60 ranks
        # are nearly flat (1/61 vs 1/62) and a 0.9 prior jumps several places. The default
        # k=20 measured better on the sample set (README, "Choosing rrf_k").
        hi = cands[0][0] or 1.0
        relevance = {c[1]: c[0] / hi for c in cands}

        rerank_scores: dict[str, float] = {}
        rr = self.reranker()
        if rr is not None:
            pool = cands[: self.cfg.reranker.pool]
            t = time.perf_counter()
            try:
                scores = run_with_timeout(lambda: rr.score(query, [c[2]["text"] for c in pool]),
                                          self.cfg.retrieval.inference_timeout_s, "rerank")
                if len(scores) > 1 and max(scores) - min(scores) < self.cfg.reranker.min_spread:
                    log.info("reranker scores flat (spread %.4f): keeping fused order", max(scores) - min(scores))
                else:
                    rerank_scores = {c[1]: float(s) for c, s in zip(pool, scores)}
            except Exception as exc:  # noqa: BLE001 - includes InferenceTimeout
                log.warning("reranker failed (%s): keeping fused order", exc)
            timings["rerank"] = time.perf_counter() - t
            if rerank_scores:
                cands = [c for c in cands if c[1] in rerank_scores]
                relevance = rerank_scores

        hits = []
        for fused, cid, row in cands:
            srcs = [name for name, ranks in (("dense", d_rank), ("sparse", s_rank)) if cid in ranks]
            hits.append(self._make_hit(row, relevance[cid], now, historical, query, fused=round(fused, 6),
                                       dense_rank=d_rank.get(cid), sparse_rank=s_rank.get(cid),
                                       rerank=rerank_scores.get(cid), sources=srcs))
        timings["total"] = time.perf_counter() - t_all
        return self._finish(query, hits, k, mode, reason, historical, timings, fast)
