import threading
import time

import numpy as np
import pytest

from hybrid_rag.embedders import HashingEmbedder
from hybrid_rag.retriever import DenseUnavailable, Retriever, run_with_timeout, InferenceTimeout


def R(cfg, **kw):
    kw.setdefault("embedder", HashingEmbedder(256))
    kw.setdefault("memory_probe", lambda: None)
    return Retriever(cfg, **kw)


def notes(res):
    return [h.note_id for h in res.hits]


def test_hybrid_uses_both_retrievers(built):
    res = R(built).retrieve("How long are nightly backups kept?", 3)
    assert res.mode == "hybrid" and res.reason == ""
    top = res.hits[0]
    assert top.note_id == "backup" and set(top.sources) == {"dense", "sparse"}
    assert top.dense_rank is not None and top.sparse_rank is not None
    assert top.relevance == pytest.approx(1.0) and top.score == pytest.approx(top.provenance["multiplier"], abs=1e-3)
    assert top.file_path == "a/backup.md" and top.section_heading == "Body"


def test_status_filter_and_historical(built):
    r = R(built)
    assert "backup-old" not in notes(r.retrieve("nightly backups kept 30 days", 8))
    hist = r.retrieve("How long were backups kept previously?", 8)
    assert hist.historical and "backup-old" in notes(hist)
    assert "backup-old" in notes(r.retrieve("nightly backups kept 30 days", 8, historical=True))


def test_freshness_breaks_a_tie_between_active_notes(built):
    q = "How much does the Team plan cost per seat?"
    off = built.copy()
    off.provenance.enabled = False
    assert set(notes(R(off).retrieve(q, 2))) == {"price-2025", "price-2026"}
    on = R(built).retrieve(q, 2)
    assert on.hits[0].note_id == "price-2026"
    fresh = {h.note_id: h.provenance["freshness"] for h in on.hits}
    # fast half-life 45 days: 31 days old -> ~0.92, 396 days old -> ~0.80 (the floor)
    assert fresh["price-2026"] == pytest.approx(0.924, abs=0.005)
    assert fresh["price-2025"] == pytest.approx(0.80, abs=0.005)


def test_trust_and_type_demote_the_chat_summary(built):
    res = R(built).retrieve("What is the API token limit in requests per minute?", 2)
    assert notes(res)[0] == "limits"
    chat = [h for h in res.hits if h.note_id == "chat"]
    if chat:
        assert chat[0].provenance["doc_type"] == 0.6 and chat[0].provenance["trust"] < 0.75


def test_fast_path_answers_exact_keys(built):
    r = R(built)
    res = r.retrieve("ADR-007", 3)
    assert res.mode == "fastpath" and res.hits[0].note_id == "adr-007"
    assert res.hits[0].sources == ["fastpath:exact-key"] and res.hits[0].score == 1.0
    mixed = r.retrieve("why did ADR-007 choose blue-green deploys and how fast is a rollback", 3)
    assert mixed.mode == "hybrid" and mixed.hits[0].note_id == "adr-007"
    assert R(built, fastpaths=[]).retrieve("ADR-007", 2).mode == "hybrid"


def test_degrades_when_dense_file_missing(built):
    built.dense_path.unlink()
    res = R(built).retrieve("How long are nightly backups kept?", 2)
    assert res.mode == "degraded" and "missing" in res.reason
    assert res.hits[0].note_id == "backup" and res.hits[0].sources == ["sparse"]
    with pytest.raises(DenseUnavailable):
        R(built).retrieve("backups", 2, mode="dense")


def test_degrades_on_embedder_mismatch(built):
    res = R(built, embedder=HashingEmbedder(64)).retrieve("backups kept", 2)
    assert res.mode == "degraded" and "re-index" in res.reason


def test_degrades_on_memory_pressure(built):
    res = R(built, memory_probe=lambda: "available RAM 3% < 10%").retrieve("backups kept", 2)
    assert res.mode == "degraded" and "memory pressure" in res.reason and res.hits


class SlowEmbedder(HashingEmbedder):
    def encode(self, texts):
        time.sleep(1.0)
        return super().encode(texts)


def test_degrades_when_embedding_overruns_budget(built):
    built.retrieval.inference_timeout_s = 0.1
    t = time.perf_counter()
    res = R(built, embedder=SlowEmbedder(256)).retrieve("backups kept", 2)
    assert res.mode == "degraded" and "exceeded" in res.reason
    assert time.perf_counter() - t < 0.9


class BoomEmbedder(HashingEmbedder):
    def encode(self, texts):
        raise RuntimeError("device lost")


def test_degrades_when_embedding_raises(built):
    res = R(built, embedder=BoomEmbedder(256)).retrieve("backups kept", 2)
    assert res.mode == "degraded" and "device lost" in res.reason


class FlatReranker:
    name = "flat"

    def score(self, query, passages):
        return [0.5] * len(passages)


class BrokenReranker:
    name = "broken"

    def score(self, query, passages):
        raise RuntimeError("onnx crashed")


class GardenLover:
    name = "garden"

    def score(self, query, passages):
        return [0.99 if "garden" in p else 0.01 for p in passages]


def test_reranker_paths(built):
    q = "How long are nightly backups kept?"
    plain = notes(R(built).retrieve(q, 3))
    assert notes(R(built, reranker=FlatReranker()).retrieve(q, 3)) == plain
    assert notes(R(built, reranker=BrokenReranker()).retrieve(q, 3)) == plain
    res = R(built, reranker=GardenLover()).retrieve("garden backups", 3)
    assert res.hits[0].note_id == "garden" and res.hits[0].rerank == pytest.approx(0.99)


def test_low_confidence_for_unanswerable(built):
    assert R(built).retrieve("parking rules for the harbour office", 3).low_confidence
    assert not R(built).retrieve("Team plan cost per seat", 3).low_confidence
    empty = R(built).retrieve("   ", 3)
    assert empty.hits == [] and empty.low_confidence


def test_modes_for_evaluation(built):
    r = R(built)
    assert r.retrieve("backups kept", 3, mode="sparse").mode == "sparse"
    dense = r.retrieve("backups kept", 3, mode="dense")
    assert dense.mode == "dense" and all(h.sources == ["dense"] for h in dense.hits)
    with pytest.raises(ValueError):
        r.retrieve("x", 1, mode="other")


def test_per_note_cap(built):
    built.retrieval.max_chunks_per_note = 1
    res = R(built).retrieve("backups plan token garden deploys", 8)
    assert len(notes(res)) == len(set(notes(res)))


def test_concurrent_queries_are_safe(built):
    r = R(built)
    errors, results = [], []

    def work():
        try:
            for _ in range(5):
                results.append(r.retrieve("How long are nightly backups kept?", 2).hits[0].note_id)
        except Exception as exc:  # pragma: no cover - the assertion reports it
            errors.append(exc)

    threads = [threading.Thread(target=work) for _ in range(6)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not errors and set(results) == {"backup"} and len(results) == 30


def test_run_with_timeout():
    assert run_with_timeout(lambda: 7, 1.0, "x") == 7
    with pytest.raises(InferenceTimeout):
        run_with_timeout(lambda: time.sleep(0.5), 0.05, "slow")
    with pytest.raises(KeyError):
        run_with_timeout(lambda: {}["k"], 1.0, "err")


def test_reloads_dense_index_after_reindex(built, embedder, vault):
    from hybrid_rag.indexer import build_index
    from tests.conftest import write_note
    r = R(built)
    assert "zeppelin" not in " ".join(h.text for h in r.retrieve("zeppelin hangar", 3).hits)
    time.sleep(0.01)
    write_note(vault, "e/zeppelin.md", id="zeppelin", body="The zeppelin hangar opens at dawn for inspections.")
    build_index(built, embedder=embedder)
    assert r.retrieve("zeppelin hangar", 3).hits[0].note_id == "zeppelin"
    assert isinstance(r.dense_index().matrix, np.ndarray)
