"""Offline demo on the bundled synthetic vault: no network, no keys, no model download.

Uses the hashing embedder and a fixed reference date (2026-11-01) so the output
is the same on every machine.
"""

from __future__ import annotations

import logging
import shutil
import tempfile
import threading
from pathlib import Path

from .cli import format_result
from .client import call_daemon
from .config import load_config
from .daemon import make_server
from .indexer import build_index
from .retriever import Retriever

REFERENCE_DATE = "2026-11-01"


def _section(title: str) -> None:
    print(f"\n=== {title} " + "=" * max(0, 74 - len(title)))


def _show(retriever: Retriever, query: str, k: int = 3, **kw: object) -> None:
    res = retriever.retrieve(query, k, **kw)  # type: ignore[arg-type]
    print(f"> {query!r}")
    print(format_result(res.to_dict()))


def run_demo(keep: str | None = None) -> int:
    # Section 8 degrades on purpose; its reason is printed in the result, so the
    # library warning would only repeat it out of order on stderr.
    logging.getLogger("hybrid_rag.retriever").setLevel(logging.ERROR)
    work = Path(keep) if keep else Path(tempfile.mkdtemp(prefix="hrag-demo-"))
    sock_dir = Path(tempfile.mkdtemp(prefix="hrag-", dir="/tmp"))   # short path: Unix socket length limit
    try:
        cfg = load_config(None, env={}, paths__data_dir=str(work / "index"),
                          embedder__backend="hashing", provenance__reference_date=REFERENCE_DATE)
        cfg.daemon.socket = str(sock_dir / "hrag.sock")
        stats = build_index(cfg, rebuild=True)
        _section("1. Index the synthetic vault")
        print(f"vault: bundled sample_vault ({stats.files_seen} notes, all fictional)")
        print(f"{stats.chunks_written} chunks, {stats.dense_rows} dense rows, embedder={stats.embedder}, "
              f"{stats.seconds:.2f}s")
        r = Retriever(cfg, memory_probe=lambda: None)

        _section("2. Hybrid fusion: dense + BM25, merged by reciprocal rank")
        _show(r, "How do I roll back a bad deploy?")

        _section("3. Provenance conflict: two price lists, both still marked active")
        no_prov = cfg.copy()
        no_prov.provenance.enabled = False
        print("-- provenance multipliers OFF:")
        _show(Retriever(no_prov, memory_probe=lambda: None), "How much does the Team plan cost per seat?", k=3)
        print("-- provenance multipliers ON (fast volatility: the 2025 list has aged a year):")
        _show(r, "How much does the Team plan cost per seat?", k=3)

        _section("4. Low-trust chat summary vs the reference page")
        _show(r, "What is the API rate limit per token?", k=3)

        _section("5. Status filter: superseded notes only come back for historical questions")
        _show(r, "How long are backups kept?", k=2)
        _show(r, "How long were nightly backups kept previously?", k=3)

        _section("6. Exact-key fast path")
        _show(r, "ADR-012", k=2)

        _section("7. Unanswerable question -> low confidence flag")
        _show(r, "What is the parking policy at the harbour office?", k=2)

        _section("8. Degraded mode: dense index removed, keyword search still answers")
        cfg.dense_path.unlink()
        _show(Retriever(cfg, memory_probe=lambda: None), "How do I roll back a bad deploy?", k=2)
        build_index(cfg, rebuild=True)   # restore for the daemon part

        _section("9. Daemon: Unix socket, newline-delimited JSON")
        srv = make_server(cfg, Retriever(cfg, memory_probe=lambda: None))
        t = threading.Thread(target=srv.serve_forever, daemon=True)
        t.start()
        try:
            print("ping:", call_daemon(cfg, {"op": "ping"}))
            reply = call_daemon(cfg, {"op": "query", "query": "When is the on-call handover?", "k": 1})
            res = reply["result"]
            top = res["hits"][0]
            print(f"query via daemon: mode={res['mode']} top={top['note_id']} "
                  f"cite={top['file_path']} > {top['section_heading']}")
        finally:
            srv.shutdown()
            srv.server_close()
        print("\ndemo OK")
        return 0
    finally:
        shutil.rmtree(sock_dir, ignore_errors=True)
        if not keep:
            shutil.rmtree(work, ignore_errors=True)
