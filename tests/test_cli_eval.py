import json
from pathlib import Path

import pytest

from hybrid_rag.cli import main
from hybrid_rag.config import SAMPLE_VAULT, load_config
from hybrid_rag.embedders import HashingEmbedder
from hybrid_rag.evaluation import (
    Variant, load_queries, markdown_table, recall_at_k, reciprocal_rank, run_eval, note_ranking,
)
from hybrid_rag.notes import parse_frontmatter

ROOT = Path(__file__).resolve().parents[1]
QUERIES = ROOT / "eval" / "questions.jsonl"


def test_metrics_hand_computed():
    ranking = note_ranking(["a", "a", "b", "c", "d"])
    assert ranking == ["a", "b", "c", "d"]
    assert reciprocal_rank(ranking, ["c"]) == pytest.approx(1 / 3)
    assert reciprocal_rank(ranking, ["z"]) == 0.0
    assert recall_at_k(ranking, ["b", "d"], 2) == 0.5
    assert recall_at_k(ranking, ["b", "d"], 4) == 1.0


def test_load_queries_validates(tmp_path):
    f = tmp_path / "q.jsonl"
    f.write_text('# comment\n{"id": "1", "query": "x", "gold": []}\n', encoding="utf-8")
    with pytest.raises(ValueError):
        load_queries(f)
    g = tmp_path / "q2.jsonl"
    g.write_text('{"id": "a", "question": "x", "gold_notes": ["n"], "tag": "t"}\n'
                 '{"id": "u", "question": "y", "gold_notes": [], "tag": "unanswerable"}\n', encoding="utf-8")
    assert [q["id"] for q in load_queries(g)] == ["a"] and load_queries(g)[0]["query"] == "x"


def test_eval_on_sample_vault_is_deterministic(tmp_path):
    cfg = load_config(None, env={}, paths__data_dir=str(tmp_path / "i"), embedder__backend="hashing",
                      provenance__reference_date="2026-11-01")
    from hybrid_rag.indexer import build_index
    emb = HashingEmbedder(512)
    build_index(cfg, embedder=emb)
    qs = load_queries(QUERIES)
    a = run_eval(cfg, qs, embedder=emb)
    b = run_eval(cfg, qs, embedder=emb)
    assert [r.metrics for r in a] == [r.metrics for r in b]
    rr = {r.name: {q["id"]: q["rr"] for q in r.per_query} for r in a}
    conflicts = ["sf-01", "sf-02", "sf-03"]   # stale runbook, chat summary vs reference, two price lists
    assert all(rr["Hybrid RRF + full provenance"][q] == 1.0 for q in conflicts)
    assert any(rr["Hybrid RRF, no provenance"][q] < 1.0 for q in conflicts)
    assert "| Configuration |" in markdown_table(a)
    one = run_eval(cfg, qs[:3], [Variant("x", mode="sparse")], embedder=emb)
    assert len(one) == 1 and len(one[0].per_query) == 3


def test_sample_vault_and_queries_are_consistent():
    ids = set()
    for path in SAMPLE_VAULT.rglob("*.md"):
        if path.name == "README.md":
            continue
        fm, _ = parse_frontmatter(path.read_text(encoding="utf-8"))
        assert fm.id and fm.doc_type and fm.generated_by and fm.volatility, path
        assert fm.reliability is not None and fm.tier is not None, path
        ids.add(fm.id)
    assert 18 <= len(ids) <= 25
    for q in load_queries(QUERIES):
        assert set(q["gold"]) <= ids, q
    assert "fictional" in (SAMPLE_VAULT / "README.md").read_text(encoding="utf-8").lower()


def test_cli_index_query_eval(tmp_path, capsys, monkeypatch):
    monkeypatch.chdir(ROOT)
    common = ["--data-dir", str(tmp_path / "idx"), "--embedder", "hashing", "--reference-date", "2026-11-01"]
    assert main(["index", *common]) == 0
    assert "indexed 21 notes" in capsys.readouterr().out
    assert main(["query", "How much does the Team plan cost per seat?", "-k", "2", "--no-daemon",
                 "--explain", *common]) == 0
    out = capsys.readouterr().out
    assert "ref-pricing-2026" in out.splitlines()[1] and "factors:" in out
    assert main(["query", "rate limit per token", "--json", "--sparse-only", *common]) == 0
    data = json.loads(capsys.readouterr().out)
    assert data["mode"] == "degraded" and data["hits"][0]["note_id"] == "ref-api-rate-limits"
    assert main(["eval", "--queries", str(QUERIES), "--json-out", str(tmp_path / "m.json"), *common]) == 0
    assert "Hybrid RRF + full provenance" in capsys.readouterr().out
    assert len(json.loads((tmp_path / "m.json").read_text())) == 5


def test_cli_errors_and_ping(tmp_path, capsys):
    assert main(["index", "--vault", str(tmp_path / "missing"), "--data-dir", str(tmp_path / "i"),
                 "--embedder", "hashing"]) == 2
    assert "vault folder not found" in capsys.readouterr().err
    import os
    os.environ["HRAG_SOCKET"] = str(tmp_path / "no.sock")
    try:
        assert main(["ping"]) == 1
    finally:
        del os.environ["HRAG_SOCKET"]


def test_demo_runs_offline(capsys):
    assert main(["demo"]) == 0
    out = capsys.readouterr().out
    assert "demo OK" in out and "mode=degraded" in out and "mode=fastpath" in out and "LOW CONFIDENCE" in out
