"""Command line: ``hrag index | query | serve | ping | eval | demo``."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from . import __version__
from .config import Config, load_config


def _cfg(args: argparse.Namespace) -> Config:
    return load_config(
        args.config,
        paths__vault=getattr(args, "vault", None),
        paths__data_dir=getattr(args, "data_dir", None),
        embedder__backend=getattr(args, "embedder", None),
        embedder__model=getattr(args, "embed_model", None),
        reranker__backend=getattr(args, "reranker", None),
        provenance__reference_date=getattr(args, "reference_date", None),
        retrieval__rrf_k=getattr(args, "rrf_k", None),
    )


def _parse_tcp(value: str | None) -> tuple[str, int] | None:
    if not value:
        return None
    host, _, port = value.rpartition(":")
    return (host or "127.0.0.1", int(port))


def format_hit_line(rank: int, h: dict, *, width: int = 100) -> list[str]:
    prov = h.get("provenance") or {}
    parts = [f"rel={h['relevance']:.3f}", f"prov=x{prov.get('multiplier', 1.0):.3f}", f"final={h['score']:.3f}"]
    src = ",".join(h.get("sources") or [])
    head = f"{rank:>2}. {h['note_id']:<26} {' '.join(parts)}  [{src}] status={h['status']}"
    cite = f"    cite: {h['file_path']} > {h['section_heading']}"
    snippet = " ".join(h["text"].split())
    snippet = snippet if len(snippet) <= width else snippet[: width - 3] + "..."
    return [head, cite, f"    {snippet}"]


def format_result(res: dict, *, explain: bool = False) -> str:
    lines = [f"mode={res['mode']}" + (f"  reason: {res['reason']}" if res.get("reason") else "")
             + ("  historical" if res.get("historical") else "")
             + ("  LOW CONFIDENCE" if res.get("low_confidence") else "")]
    if not res["hits"]:
        lines.append("  (no hits)")
    for i, h in enumerate(res["hits"], 1):
        lines += format_hit_line(i, h)
        if explain:
            p = h.get("provenance") or {}
            lines.append("    factors: " + " ".join(f"{k}={v}" for k, v in p.items() if k != "multiplier")
                         + f"  ranks: dense={h.get('dense_rank')} sparse={h.get('sparse_rank')}"
                         + f"  coverage={h.get('coverage')}")
    return "\n".join(lines)


def cmd_index(args: argparse.Namespace) -> int:
    from .indexer import build_index
    cfg = _cfg(args)
    stats = build_index(cfg, rebuild=args.rebuild)
    print(f"indexed {stats.files_indexed} notes ({stats.files_unchanged} unchanged, {stats.files_deleted} removed), "
          f"{stats.chunks_written} chunks written, {stats.dense_rows} dense rows, embedder={stats.embedder}, "
          f"{stats.seconds:.2f}s -> {cfg.paths.data_dir}")
    return 0


def cmd_query(args: argparse.Namespace) -> int:
    from .client import query
    cfg = _cfg(args)
    if args.no_provenance:
        cfg.provenance.enabled = False
    historical = True if args.historical else None
    if args.sparse_only or args.no_daemon or args.no_provenance:
        from .retriever import Retriever
        res = Retriever(cfg).retrieve(args.query, args.k, historical=historical, force_sparse=args.sparse_only)
        out = res.to_dict()
        out["via"] = "in-process"
    else:
        out = query(cfg, args.query, args.k, historical=historical, full_fallback=True, tcp=_parse_tcp(args.tcp))
    if args.json:
        print(json.dumps(out, indent=2, ensure_ascii=False))
    else:
        print(format_result(out, explain=args.explain))
    return 0


def cmd_serve(args: argparse.Namespace) -> int:
    from .daemon import serve
    serve(_cfg(args), tcp=_parse_tcp(args.tcp))
    return 0


def cmd_ping(args: argparse.Namespace) -> int:
    from .client import DaemonError, call_daemon
    try:
        reply = call_daemon(_cfg(args), {"op": "ping"}, tcp=_parse_tcp(args.tcp))
    except DaemonError as exc:
        print(f"no daemon: {exc}")
        return 1
    print(json.dumps(reply))
    return 0


def cmd_eval(args: argparse.Namespace) -> int:
    from .evaluation import load_queries, markdown_table, run_eval
    from .indexer import build_index
    cfg = _cfg(args)
    if not cfg.provenance.reference_date:
        cfg.provenance.reference_date = "2026-11-01"   # fixed clock => reproducible freshness
    from .embedders import make_embedder
    emb = make_embedder(cfg.embedder)
    build_index(cfg, embedder=emb)
    results = run_eval(cfg, load_queries(args.queries), embedder=emb)
    print(f"embedder={emb.name} reranker={cfg.reranker.backend} rrf_k={cfg.retrieval.rrf_k} "
          f"reference_date={cfg.provenance.reference_date} queries={args.queries}")
    print(markdown_table(results))
    if args.per_query:
        for r in results:
            print(f"\n## {r.name}")
            for q in r.per_query:
                mark = "ok " if q["rr"] == 1.0 else ("~  " if q["rr"] > 0 else "MISS")
                print(f"  {mark} {q['id']:<4} rr={q['rr']:.2f} gold={q['gold']} top3={q['top'][:3]}")
    if args.json_out:
        Path(args.json_out).write_text(json.dumps([{"name": r.name, "metrics": r.metrics, "per_query": r.per_query}
                                                   for r in results], indent=2), encoding="utf-8")
    return 0


def cmd_demo(args: argparse.Namespace) -> int:
    from .demo import run_demo
    return run_demo(keep=args.keep)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="hrag", description="Hybrid RAG with provenance ranking over markdown notes")
    p.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    p.add_argument("--config", help="TOML config file (default: $HRAG_CONFIG or ./hrag.toml)")
    p.add_argument("-v", "--verbose", action="store_true")
    sub = p.add_subparsers(dest="cmd", required=True)

    def common(sp: argparse.ArgumentParser) -> None:
        sp.add_argument("--vault", help="folder of markdown notes")
        sp.add_argument("--data-dir", help="index directory")
        sp.add_argument("--embedder", help="hashing | sentence-transformers")
        sp.add_argument("--embed-model", help="model name for sentence-transformers")
        sp.add_argument("--reranker", help="none | overlap | cross-encoder | flashrank")
        sp.add_argument("--reference-date", help="YYYY-MM-DD 'today' for the freshness clock")
        sp.add_argument("--rrf-k", type=int, help="reciprocal-rank fusion constant")

    sp = sub.add_parser("index", help="build or update the index")
    common(sp)
    sp.add_argument("--rebuild", action="store_true", help="drop the index and rebuild from scratch")
    sp.set_defaults(fn=cmd_index)

    sp = sub.add_parser("query", help="retrieve notes for a query")
    common(sp)
    sp.add_argument("query")
    sp.add_argument("-k", type=int, default=5)
    sp.add_argument("--historical", action="store_true", help="include superseded/archived notes, no freshness decay")
    sp.add_argument("--sparse-only", action="store_true", help="keyword search only (degraded mode)")
    sp.add_argument("--no-provenance", action="store_true", help="switch the provenance multipliers off")
    sp.add_argument("--no-daemon", action="store_true", help="always run in-process")
    sp.add_argument("--tcp", help="daemon address host:port instead of the Unix socket")
    sp.add_argument("--explain", action="store_true", help="show every provenance factor and the ranks")
    sp.add_argument("--json", action="store_true")
    sp.set_defaults(fn=cmd_query)

    sp = sub.add_parser("serve", help="run the retrieval daemon")
    common(sp)
    sp.add_argument("--tcp", help="listen on host:port (localhost) instead of a Unix socket")
    sp.set_defaults(fn=cmd_serve)

    sp = sub.add_parser("ping", help="check the daemon")
    sp.add_argument("--tcp")
    sp.set_defaults(fn=cmd_ping)

    sp = sub.add_parser("eval", help="recall@k and MRR on labelled queries")
    common(sp)
    sp.add_argument("--queries", default="eval/questions.jsonl")
    sp.add_argument("--per-query", action="store_true")
    sp.add_argument("--json-out")
    sp.set_defaults(fn=cmd_eval)

    sp = sub.add_parser("demo", help="offline demo on the bundled synthetic vault")
    sp.add_argument("--keep", help="keep the demo index in this folder")
    sp.set_defaults(fn=cmd_demo)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING,
                        format="%(levelname)s %(name)s: %(message)s")
    try:
        return args.fn(args)
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
