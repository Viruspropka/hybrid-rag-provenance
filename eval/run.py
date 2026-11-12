"""Run the retrieval and answer evaluation and write a run folder.

    python -m eval.run --seed 42            # what `make eval` runs: offline, no keys

Writes ``eval/runs/<UTC timestamp>/`` with:

* ``run.log``            full log of the run (config, every question, the tables)
* ``per_question.jsonl`` one line per (configuration, question) and one per answer
* ``metrics.json``       every aggregate number; deterministic for a given code + data + seed
* ``environment.json``   commit, date, machine, versions, wall time (not deterministic)

and then regenerates ``RESULTS.md`` and the README evaluation table from the
committed runs (``eval/report.py``). The default profile (hashing embedder,
lexical overlap reranker) needs no network: a socket guard makes any outbound
connection attempt fail the run.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import platform
import random
import socket
import sqlite3
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:          # run from a checkout without `pip install -e .`
    sys.path.insert(0, str(ROOT / "src"))

from hybrid_rag.config import SAMPLE_VAULT, load_config  # noqa: E402
from hybrid_rag.embedders import make_embedder  # noqa: E402
from hybrid_rag.indexer import build_index  # noqa: E402
from hybrid_rag.rerankers import make_reranker  # noqa: E402
from hybrid_rag.retriever import Retriever  # noqa: E402

from . import answer as ans_mod  # noqa: E402
from .metrics import collapse_to_notes, first_gold_rank, mean, mrr_at_k, recall_at_k  # noqa: E402

EVAL_DIR = ROOT / "eval"
QUESTIONS = EVAL_DIR / "questions.jsonl"
MANIFEST = EVAL_DIR / "MANIFEST"
RUNS = EVAL_DIR / "runs"
TAGS = ("keyword-heavy", "paraphrase", "multi-hop", "stale-vs-fresh", "unanswerable")
KS = (1, 5, 10)
DEPTH = 10                # MRR@10, recall@10
FETCH_CHUNKS = 40         # chunks fetched per query before collapsing to notes
REFERENCE_DATE = "2026-11-01"
ANSWER_CONFIG = "hybrid_prov"

log = logging.getLogger("eval")


@dataclass(frozen=True)
class Variant:
    key: str
    name: str
    mode: str
    provenance: bool = False
    rrf_k: int = 20
    rerank: bool = False


VARIANTS = (
    Variant("fts5", "FTS5 only (BM25)", "sparse"),
    Variant("dense", "Dense only", "dense"),
    Variant("hybrid", "Hybrid RRF (k=20)", "hybrid"),
    Variant("hybrid_prov", "Hybrid RRF (k=20) + provenance", "hybrid", provenance=True),
    Variant("hybrid_prov_k60", "Hybrid RRF (k=60) + provenance", "hybrid", provenance=True, rrf_k=60),
    Variant("hybrid_prov_rerank", "Hybrid RRF (k=20) + provenance + reranker", "hybrid", provenance=True,
            rerank=True),
)


# ----------------------------------------------------------------- data
def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def manifest_hash(manifest: Path = MANIFEST, name: str = "questions.jsonl") -> str | None:
    if not manifest.is_file():
        return None
    for line in manifest.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) == 2 and parts[1] == name:
            return parts[0]
    return None


def load_questions(path: Path) -> list[dict]:
    items = []
    for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        q = json.loads(line)
        for key in ("id", "question", "gold_notes", "gold_answer", "required_citations", "tag"):
            if key not in q:
                raise ValueError(f"{path}:{n}: missing '{key}'")
        if q["tag"] not in TAGS:
            raise ValueError(f"{path}:{n}: unknown tag {q['tag']!r}")
        if (q["tag"] == "unanswerable") != (not q["gold_notes"]):
            raise ValueError(f"{path}:{n}: unanswerable items and only they have no gold notes")
        if not set(q["required_citations"]) <= set(q["gold_notes"]):
            raise ValueError(f"{path}:{n}: required citations must be gold notes")
        items.append(q)
    ids = [q["id"] for q in items]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate question ids")
    return items


def subset(items: list[dict], n: int, seed: int) -> list[dict]:
    """``n`` questions drawn with ``random.Random(seed)``, kept in file order."""
    if n <= 0 or n >= len(items):
        return items
    pick = set(random.Random(seed).sample(range(len(items)), n))
    return [q for i, q in enumerate(items) if i in pick]


# ------------------------------------------------------------ helpers
class NetworkBlocked(RuntimeError):
    pass


def block_network() -> None:
    """Make every non-loopback connection attempt raise: proves the default run is offline."""
    real_connect = socket.socket.connect

    def guarded(self, address):  # type: ignore[no-untyped-def]
        host = address[0] if isinstance(address, tuple) else address
        if isinstance(host, str) and (host in ("localhost", "::1") or host.startswith("127.") or "/" in host):
            return real_connect(self, address)
        raise NetworkBlocked(f"network access attempted during an offline eval run: {address!r}")

    socket.socket.connect = guarded  # type: ignore[method-assign]
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")


def git(*args: str) -> str:
    """Raw stdout of a git command (not stripped: porcelain lines start with a space)."""
    try:
        return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, timeout=10,
                              check=True).stdout
    except (OSError, subprocess.SubprocessError):
        return ""


def code_state() -> dict:
    """HEAD commit and uncommitted changes outside the generated outputs."""
    generated = ("eval/runs/", "RESULTS.md", "README.md")
    dirty = [line[3:] for line in git("status", "--porcelain").splitlines()
             if len(line) > 3 and not line[3:].startswith(generated)]
    return {"commit": git("rev-parse", "HEAD").strip() or "unknown", "uncommitted_changes": dirty}


def machine() -> dict:
    cpu = ""
    if sys.platform == "darwin":
        try:
            cpu = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True,
                                 text=True, timeout=5).stdout.strip()
        except (OSError, subprocess.SubprocessError):
            cpu = ""
    elif Path("/proc/cpuinfo").is_file():
        for line in Path("/proc/cpuinfo").read_text(errors="replace").splitlines():
            if line.lower().startswith("model name"):
                cpu = line.split(":", 1)[1].strip()
                break
    return {  # no hostname, no user name, no paths
        "cpu": cpu or platform.processor() or platform.machine(),
        "cpu_count": os.cpu_count(),
        "os": platform.platform(),
        "python": platform.python_version(),
        "sqlite": sqlite3.sqlite_version,
    }


def versions(names: list[str]) -> dict:
    from importlib import metadata
    out = {}
    for n in names:
        try:
            out[n] = metadata.version(n)
        except metadata.PackageNotFoundError:
            pass
    return out


# --------------------------------------------------------------- core
def evaluate(args: argparse.Namespace, run_dir: Path) -> dict:
    qpath = Path(args.questions)
    items = load_questions(qpath)
    qsha = sha256_file(qpath)
    expected = manifest_hash()
    if qpath.resolve() == QUESTIONS.resolve() and expected != qsha:
        raise SystemExit(f"eval/questions.jsonl does not match eval/MANIFEST ({qsha} vs {expected}); "
                         "the frozen set was edited")
    items = subset(items, args.subset, args.seed)
    random.seed(args.seed)
    log.info("questions: %s (%d items, sha256 %s), seed %d", qpath.name, len(items), qsha, args.seed)

    with tempfile.TemporaryDirectory(prefix="hrag-eval-") as tmp:
        cfg = load_config(None, env={}, paths__vault=str(SAMPLE_VAULT), paths__data_dir=tmp,
                          embedder__backend=args.embedder, embedder__model=args.embed_model,
                          embedder__fallback="", embedder__device=args.device,
                          provenance__reference_date=REFERENCE_DATE)
        cfg.fastpath.exact_key_pattern = ""        # measure the retrievers, not the exact-key plugin
        emb = make_embedder(cfg.embedder)
        t0 = time.perf_counter()
        stats = build_index(cfg, embedder=emb)
        log.info("index: %s with %s in %.2fs", stats, emb.name, time.perf_counter() - t0)

        rcfg = cfg.copy()
        rcfg.reranker.backend = args.reranker
        if args.reranker_model:
            rcfg.reranker.model = args.reranker_model
        reranker = make_reranker(rcfg.reranker)
        rr_name = reranker.name if reranker else "none"
        variants = [v for v in VARIANTS if not v.rerank or reranker is not None]

        rows: list[dict] = []
        retrieval: dict[str, dict] = {}
        answers: list[dict] = []
        sweep_rows: dict[tuple, list[dict]] = {cell: [] for cell in ans_mod.SWEEP}
        for v in variants:
            vcfg = cfg.copy()
            vcfg.provenance.enabled = v.provenance
            vcfg.provenance.status_filter = v.provenance
            vcfg.retrieval.rrf_k = v.rrf_k
            vcfg.reranker.backend = rcfg.reranker.backend if v.rerank else "none"
            retr = Retriever(vcfg, embedder=emb, reranker=reranker if v.rerank else None,
                             memory_probe=lambda: None)
            per_q = []
            for q in items:
                res = retr.retrieve(q["question"], FETCH_CHUNKS, mode=v.mode)
                ranking = collapse_to_notes(h.note_id for h in res.hits)[:DEPTH]
                row = {"kind": "retrieval", "config": v.key, "id": q["id"], "tag": q["tag"],
                       "gold": q["gold_notes"], "top": ranking, "mode": res.mode,
                       "historical": res.historical, "low_confidence": res.low_confidence}
                if q["gold_notes"]:
                    for k in KS:
                        row[f"recall@{k}"] = recall_at_k(ranking, q["gold_notes"], k)
                    row["mrr@10"] = mrr_at_k(ranking, q["gold_notes"], DEPTH)
                    row["first_gold_rank"] = first_gold_rank(ranking, q["gold_notes"])
                per_q.append(row)
                if v.key == ANSWER_CONFIG:
                    passages = [ans_mod.Passage(h.note_id, h.chunk_id, h.text) for h in res.hits]
                    a = ans_mod.answer(q["question"], passages, res.low_confidence)
                    used = passages[:ans_mod.CONTEXT_HITS]
                    chk = ans_mod.check(a, used, q["gold_notes"], q["required_citations"], q["gold_answer"])
                    answers.append({"kind": "answer", "config": v.key, "id": q["id"], "tag": q["tag"],
                                    "question": q["question"], "gold": q["gold_notes"],
                                    "gold_answer": q["gold_answer"], "answer": a, "check": chk})
                    for cell in ans_mod.SWEEP:
                        sa = ans_mod.answer(q["question"], passages, res.low_confidence,
                                            abstain_below=cell[0], use_retriever_flag=cell[1])
                        sweep_rows[cell].append({"tag": q["tag"], "check": ans_mod.check(
                            sa, used, q["gold_notes"], q["required_citations"], q["gold_answer"])})
            rows += per_q
            retrieval[v.key] = {"name": v.name, "overall": aggregate(per_q),
                                "by_tag": {t: aggregate([r for r in per_q if r["tag"] == t])
                                           for t in TAGS if t != "unanswerable"}}
            log.info("%-45s %s", v.name, fmt(retrieval[v.key]["overall"]))
            for r in per_q:
                log.debug("  %s %-6s %s", v.key, r["id"], json.dumps({k: r[k] for k in r if k not in ("kind",)}))

    answer_metrics = {"config": ANSWER_CONFIG, "overall": answer_aggregate(answers),
                      "by_tag": {t: answer_aggregate([a for a in answers if a["tag"] == t]) for t in TAGS},
                      "thresholds": {"context_hits": ans_mod.CONTEXT_HITS, "max_sentences": ans_mod.MAX_SENTENCES,
                                     "abstain_below_coverage": ans_mod.ABSTAIN_BELOW,
                                     "second_sentence_min_new_terms": ans_mod.SECOND_MIN_NEW_TERMS,
                                     "support_threshold": ans_mod.SUPPORT_THRESHOLD}}
    answer_metrics["sweep"] = [{"abstain_below": cell[0], "retriever_flag": cell[1],
                                **answer_aggregate(rows_)} for cell, rows_ in sweep_rows.items()]
    for a in answers:
        log.debug("  answer %-6s %s", a["id"], json.dumps({"answer": a["answer"], "check": a["check"]}))
    log.info("answers (%s): %s", ANSWER_CONFIG, fmt(answer_metrics["overall"]))

    with (run_dir / "per_question.jsonl").open("w", encoding="utf-8") as f:
        for r in rows + answers:
            f.write(json.dumps(r, sort_keys=True) + "\n")
    profile = f"{emb.name}+{rr_name}"
    metrics = {
        "schema": 1, "profile": profile, "seed": args.seed,
        "questions_file": "eval/questions.jsonl" if qpath.resolve() == QUESTIONS.resolve() else qpath.name,
        "questions_sha256": qsha, "subset": args.subset, "n_questions": len(items),
        "n_answerable": sum(1 for q in items if q["gold_notes"]),
        "n_unanswerable": sum(1 for q in items if not q["gold_notes"]),
        "tag_counts": {t: sum(1 for q in items if q["tag"] == t) for t in TAGS},
        "reference_date": REFERENCE_DATE, "embedder": emb.name, "reranker": rr_name,
        "fastpath": "off", "depth": DEPTH, "fetch_chunks": FETCH_CHUNKS,
        "configs": [{"key": v.key, "name": v.name, "mode": v.mode, "provenance": v.provenance,
                     "status_filter": v.provenance, "rrf_k": v.rrf_k if v.mode == "hybrid" else None,
                     "reranker": rr_name if v.rerank else "none"} for v in variants],
        "retrieval": retrieval, "answer": answer_metrics,
    }
    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return metrics


def aggregate(rows: list[dict]) -> dict:
    scored = [r for r in rows if r["gold"]]
    out: dict = {"n": len(scored)}
    for k in KS:
        out[f"recall@{k}"] = mean([r[f"recall@{k}"] for r in scored])
    out["mrr@10"] = mean([r["mrr@10"] for r in scored])
    return out


def answer_aggregate(answers: list[dict]) -> dict:
    ans = [a["check"] for a in answers if a["check"]["answerable"]]
    una = [a["check"] for a in answers if not a["check"]["answerable"]]
    answered = [c for c in ans if not c["abstained"]]
    out: dict = {"n_answerable": len(ans), "n_unanswerable": len(una)}
    if ans:
        out |= {
            "answered_rate": len(answered) / len(ans),
            "false_abstention_rate": 1 - len(answered) / len(ans),
            "citation_recall": mean([c["citation_recall"] for c in ans]),
            "answer_term_recall": mean([c["answer_term_recall"] for c in ans]),
        }
    if answered:
        out |= {
            "citation_precision": mean([c["citation_precision"] for c in answered]),
            "citations_in_retrieved_rate": mean([1.0 if c["citations_in_retrieved"] else 0.0 for c in answered]),
            "faithfulness_proxy": mean([c["faithfulness"] for c in answered]),
        }
    if una:
        out["unanswerable_abstention_rate"] = mean([1.0 if c["correct_abstention"] else 0.0 for c in una])
    return out


def fmt(m: dict) -> str:
    return " ".join(f"{k}={v:.3f}" if isinstance(v, float) else f"{k}={v}" for k, v in m.items())


# ---------------------------------------------------------------- main
def parse_args(argv: list[str] | None) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="python -m eval.run", description=__doc__.split("\n\n")[0])
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--questions", default=str(QUESTIONS))
    p.add_argument("--subset", type=int, default=0, help="evaluate N seeded random questions (0 = all)")
    p.add_argument("--embedder", default="hashing", help="hashing (default, offline) or sentence-transformers")
    p.add_argument("--embed-model", default="BAAI/bge-m3", help="model for --embedder sentence-transformers")
    p.add_argument("--device", default="cpu")
    p.add_argument("--reranker", default="overlap", help="overlap (default, lexical stand-in), cross-encoder, none")
    p.add_argument("--reranker-model", default="")
    p.add_argument("--out-dir", default=str(RUNS))
    p.add_argument("--no-report", action="store_true", help="do not regenerate RESULTS.md and README")
    p.add_argument("--allow-network", action="store_true",
                   help="allow model downloads (only meaningful for sentence-transformers)")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if not args.allow_network:
        block_network()
    started = datetime.now(tz=timezone.utc)
    run_id = started.strftime("%Y%m%dT%H%M%SZ")
    run_dir = Path(args.out_dir) / run_id
    n = 1
    while run_dir.exists():
        n += 1
        run_dir = Path(args.out_dir) / f"{run_id}-{n}"
    run_dir.mkdir(parents=True)

    root_log = logging.getLogger()
    root_log.setLevel(logging.DEBUG)
    fh = logging.FileHandler(run_dir / "run.log", encoding="utf-8")
    fh.setFormatter(logging.Formatter("%(levelname)s %(name)s: %(message)s"))
    sh = logging.StreamHandler(sys.stdout)
    sh.setLevel(logging.INFO)
    sh.setFormatter(logging.Formatter("%(message)s"))
    root_log.addHandler(fh)
    root_log.addHandler(sh)
    for noisy in ("urllib3", "filelock", "huggingface_hub", "sentence_transformers", "transformers"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    try:
        cmd = "python -m eval.run " + " ".join(argv if argv is not None else sys.argv[1:])
        state = code_state()
        log.info("run %s: %s", run_dir.name, cmd.strip())
        log.info("code commit %s, uncommitted changes: %s", state["commit"], state["uncommitted_changes"] or "none")
        t0 = time.perf_counter()
        metrics = evaluate(args, run_dir)
        wall = time.perf_counter() - t0
        env = {"run_id": run_dir.name, "started_utc": started.isoformat(timespec="seconds"),
               "command": cmd.strip(), **state, "machine": machine(), "wall_seconds": round(wall, 2),
               "packages": versions(["numpy", "sentence-transformers", "torch", "transformers"])}
        (run_dir / "environment.json").write_text(json.dumps(env, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        log.info("wall time %.2fs; run folder %s", wall, run_dir.relative_to(ROOT) if run_dir.is_relative_to(ROOT)
                 else run_dir)
        if not args.no_report:
            from .report import regenerate
            for path in regenerate(ROOT):
                log.info("regenerated %s", path.relative_to(ROOT))
        log.info("profile %s: %d questions", metrics["profile"], metrics["n_questions"])
    finally:
        root_log.removeHandler(fh)
        root_log.removeHandler(sh)
        fh.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
