"""The evaluation harness in eval/: metric code, answer checker, frozen set, a 5-question run."""

import hashlib
import json
from pathlib import Path

import pytest

from eval import answer as A
from eval import run as R
from eval.metrics import collapse_to_notes, first_gold_rank, mean, mrr_at_k, recall_at_k

ROOT = Path(__file__).resolve().parents[1]


# --------------------------------------------------------------- metrics, hand-computed
def test_recall_and_mrr_on_a_toy_ranking():
    ranking = collapse_to_notes(["a", "a", "b", "c", "b", "d", "e", "f", "g", "h", "i", "j", "k"])
    assert ranking == ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k"]
    # single gold note at rank 3
    assert recall_at_k(ranking, ["c"], 1) == 0.0
    assert recall_at_k(ranking, ["c"], 5) == 1.0
    assert mrr_at_k(ranking, ["c"], 10) == pytest.approx(1 / 3)
    # two gold notes at ranks 2 and 7: recall@1 0, @5 0.5, @10 1; MRR uses the first one
    assert recall_at_k(ranking, ["b", "g"], 1) == 0.0
    assert recall_at_k(ranking, ["b", "g"], 5) == 0.5
    assert recall_at_k(ranking, ["b", "g"], 10) == 1.0
    assert mrr_at_k(ranking, ["g", "b"], 10) == 0.5
    # gold at rank 11 is outside MRR@10 and recall@10
    assert mrr_at_k(ranking, ["k"], 10) == 0.0
    assert recall_at_k(ranking, ["k"], 10) == 0.0
    assert first_gold_rank(ranking, ["k"]) == 11
    assert first_gold_rank(ranking, ["zz"]) is None
    # averaged over three toy queries: MRR = (1 + 1/2 + 0) / 3
    assert mean([mrr_at_k(["x"], ["x"]), mrr_at_k(["y", "x"], ["x"]), mrr_at_k(["y"], ["x"])]) == pytest.approx(0.5)


def test_metric_guards():
    with pytest.raises(ValueError):
        recall_at_k(["a"], [], 1)
    with pytest.raises(ValueError):
        recall_at_k(["a"], ["a"], 0)
    assert mean([]) == 0.0


# ------------------------------------------------------------------ answerer and checker
PASSAGES = [
    A.Passage("ref-limits", "c1", "# Limits\n\n## Per token\n\nEach API token may send 600 requests per minute. "
              "Bursts of 50 requests are allowed."),
    A.Passage("chat", "c2", "In the chat we said the limit is probably 100 requests per minute."),
]


def test_extractive_answer_cites_its_source_and_is_faithful():
    a = A.answer("How many requests per minute may an API token send?", PASSAGES, low_confidence=False)
    assert not a["abstained"] and a["citations"][0] == "ref-limits"
    assert a["sentences"][0]["text"].startswith("Each API token may send 600")
    c = A.check(a, PASSAGES, ["ref-limits"], ["ref-limits"], "600 requests per minute")
    assert c["citations_in_retrieved"] and c["citation_recall"] == 1.0 and c["faithfulness"] == 1.0
    assert c["answer_term_recall"] == 1.0


def test_answerer_abstains():
    assert A.answer("What is the office Wi-Fi password?", PASSAGES, low_confidence=False)["abstained"]
    assert A.answer("API token requests per minute", PASSAGES, low_confidence=True)["abstained"]
    a = A.answer("API token requests per minute", PASSAGES, low_confidence=True, use_retriever_flag=False)
    assert not a["abstained"]


def test_checker_flags_unfaithful_and_wrong_citations():
    bad = {"abstained": False, "citations": ["ref-limits", "elsewhere"], "sentences": [
        {"text": "Each API token may send 600 requests per minute.", "note_id": "ref-limits"},
        {"text": "Enterprise customers receive unlimited throughput guarantees.", "note_id": "elsewhere"}]}
    c = A.check(bad, PASSAGES, ["ref-limits"], ["ref-limits"], "600 requests per minute")
    assert c["citations_in_retrieved"] is False          # "elsewhere" was never retrieved
    assert c["citation_precision"] == 0.5                 # 1 of 2 cited notes is gold
    assert c["citation_recall"] == 1.0
    assert c["sentence_support"] == [1.0, 0.0]            # copied sentence vs invented sentence
    assert c["faithfulness"] == 0.5
    # content terms: each, api, token, may, send, 600, requests, minute ("per" is a stopword);
    # the chat passage contains only "requests" and "minute" -> 2 / 8, below the 0.6 threshold
    assert A.sentence_support("Each API token may send 600 requests per minute.", PASSAGES[1].text) == 0.25
    wrong = {"abstained": False, "citations": ["chat"], "sentences": [
        {"text": "Each API token may send 600 requests per minute.", "note_id": "chat"}]}
    cw = A.check(wrong, PASSAGES, ["ref-limits"], ["ref-limits"], "600 requests per minute")
    assert cw["citation_precision"] == 0.0 and cw["citation_recall"] == 0.0 and cw["faithfulness"] == 0.0
    una = A.check(bad, PASSAGES, [], [], "")
    assert una["correct_abstention"] is False
    assert A.check({"abstained": True, "citations": []}, PASSAGES, [], [], "")["correct_abstention"] is True


def test_split_sentences_drops_headings():
    assert A.split_sentences("# T\n\n## S\n\nOne two. Three four? `cmd` five.") == ["One two.", "Three four?", "`cmd` five."]


# ------------------------------------------------------------------ frozen question set
def test_question_set_is_frozen_and_well_formed():
    data = (ROOT / "eval" / "questions.jsonl").read_bytes()
    assert R.manifest_hash() == hashlib.sha256(data).hexdigest()
    items = R.load_questions(ROOT / "eval" / "questions.jsonl")
    assert 40 <= len(items) <= 60
    tags = [q["tag"] for q in items]
    assert set(tags) == set(R.TAGS) and tags.count("unanswerable") >= 5
    note_ids = R_note_ids()
    for q in items:
        assert set(q["gold_notes"]) <= note_ids, q["id"]
        if q["tag"] == "multi-hop":
            assert len(q["gold_notes"]) >= 2
        if q["gold_notes"]:
            assert q["gold_answer"] and q["required_citations"]


def R_note_ids() -> set[str]:
    from hybrid_rag.config import SAMPLE_VAULT
    from hybrid_rag.notes import parse_frontmatter
    return {parse_frontmatter(p.read_text(encoding="utf-8"))[0].id
            for p in SAMPLE_VAULT.rglob("*.md") if p.name != "README.md"}


def test_load_questions_rejects_bad_items(tmp_path):
    f = tmp_path / "q.jsonl"
    f.write_text(json.dumps({"id": "x", "question": "q", "gold_notes": [], "gold_answer": "",
                             "required_citations": [], "tag": "paraphrase"}) + "\n", encoding="utf-8")
    with pytest.raises(ValueError):
        R.load_questions(f)


# ------------------------------------------------------------------ 5-question run
def test_eval_runs_on_a_seeded_5_question_subset(tmp_path):
    out = tmp_path / "runs"
    assert R.main(["--seed", "42", "--subset", "5", "--out-dir", str(out), "--no-report"]) == 0
    assert R.main(["--seed", "42", "--subset", "5", "--out-dir", str(out), "--no-report"]) == 0
    runs = sorted(out.iterdir())
    assert len(runs) == 2
    for d in runs:
        assert {p.name for p in d.iterdir()} == {"run.log", "per_question.jsonl", "metrics.json", "environment.json"}
    m = json.loads((runs[0] / "metrics.json").read_text())
    assert m["n_questions"] == 5 and m["seed"] == 42 and m["profile"] == "hashing-512+overlap"
    assert [c["key"] for c in m["configs"]] == [v.key for v in R.VARIANTS]
    for v in m["retrieval"].values():
        for key in ("recall@1", "recall@5", "recall@10", "mrr@10"):
            assert 0.0 <= v["overall"][key] <= 1.0
        assert v["overall"]["recall@1"] <= v["overall"]["recall@5"] <= v["overall"]["recall@10"]
    # deterministic: same code, data and seed -> byte-identical metrics and per-question output
    for name in ("metrics.json", "per_question.jsonl"):
        assert (runs[0] / name).read_bytes() == (runs[1] / name).read_bytes()
    # the same seed picks the same subset; per-question rows agree with the aggregate
    rows = [json.loads(x) for x in (runs[0] / "per_question.jsonl").read_text().splitlines()]
    fts = [r for r in rows if r["kind"] == "retrieval" and r["config"] == "fts5" and r["gold"]]
    assert m["retrieval"]["fts5"]["overall"]["mrr@10"] == pytest.approx(mean([r["mrr@10"] for r in fts]))
    env = json.loads((runs[0] / "environment.json").read_text())
    assert "hostname" not in json.dumps(env).lower() and env["machine"]["python"]


def test_code_state_parses_porcelain(monkeypatch):
    monkeypatch.setattr(R, "git", lambda *a: " M README.md\n M eval/run.py\n?? eval/runs/x/\n" if a[0] == "status"
                        else "abc123\n")
    assert R.code_state() == {"commit": "abc123", "uncommitted_changes": ["eval/run.py"]}


def test_subset_is_seeded():
    items = [{"id": str(i)} for i in range(20)]
    assert R.subset(items, 5, 42) == R.subset(items, 5, 42)
    assert R.subset(items, 0, 42) == items
