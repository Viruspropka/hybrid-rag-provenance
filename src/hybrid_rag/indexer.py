"""Walk a folder of markdown notes and (re)build the hybrid index incrementally.

Unchanged notes (same sha1) are skipped, changed notes are re-chunked and
re-embedded, deleted notes are evicted. Switching the embedder forces a full
rebuild, because vectors from two models cannot share one space.
"""

from __future__ import annotations

import fcntl
import fnmatch
import hashlib
import json
import logging
import os
import time
from dataclasses import dataclass
from pathlib import Path

from . import store
from .config import Config
from .embedders import Embedder, make_embedder
from .notes import chunk_note

log = logging.getLogger(__name__)

EXCLUDED_DIRS = frozenset({".git", ".obsidian", ".trash", "_removed_backups", "node_modules"})


@dataclass
class IndexStats:
    files_seen: int = 0
    files_indexed: int = 0
    files_unchanged: int = 0
    files_deleted: int = 0
    chunks_written: int = 0
    dense_rows: int = 0
    embedder: str = ""
    seconds: float = 0.0


def iter_notes(vault: Path, exclude: list[str] | None = None):
    """Yield (absolute path, vault-relative posix path) for every ``.md`` note."""
    for path in sorted(vault.rglob("*.md")):
        rel = path.relative_to(vault)
        if any(fnmatch.fnmatchcase(rel.as_posix(), pattern) for pattern in exclude or []):
            continue
        if any(part in EXCLUDED_DIRS or part.startswith(".") for part in rel.parts[:-1]):
            continue
        if ".sync-conflict-" in path.name:
            continue
        yield path, rel.as_posix()


class _Lock:
    """Single-writer lock so two index runs never interleave."""

    def __init__(self, path: Path) -> None:
        self._path = path
        self._fh = None

    def __enter__(self) -> "_Lock":
        store.ensure_data_dir(self._path.parent)
        fd = os.open(self._path, os.O_WRONLY | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0), 0o600)
        self._fh = os.fdopen(fd, "w")
        try:
            fcntl.flock(self._fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            self._fh.close()
            raise RuntimeError("another index run holds the lock") from exc
        return self

    def __exit__(self, *exc: object) -> None:
        if self._fh:
            fcntl.flock(self._fh, fcntl.LOCK_UN)
            self._fh.close()


def build_index(cfg: Config, *, rebuild: bool = False, embedder: Embedder | None = None) -> IndexStats:
    vault = cfg.paths.vault
    if not vault.is_dir():
        raise FileNotFoundError(f"vault folder not found: {vault}")
    t0 = time.perf_counter()
    stats = IndexStats()
    with _Lock(cfg.paths.data_dir / "index.lock"):
        if rebuild:
            for p in (cfg.index_path, cfg.dense_path, cfg.dense_ids_path):
                p.unlink(missing_ok=True)
            for suffix in ("-wal", "-shm"):
                Path(str(cfg.index_path) + suffix).unlink(missing_ok=True)
        emb = embedder or make_embedder(cfg.embedder)
        stats.embedder = emb.name
        conn = store.connect(cfg.index_path)
        try:
            previous = store.get_meta(conn, "embedder")
            if previous and previous != emb.name:
                log.warning("embedder changed (%s -> %s): re-embedding everything", previous, emb.name)
                conn.executescript("DELETE FROM chunks; DELETE FROM chunks_fts; DELETE FROM vectors; DELETE FROM files;")
            known = {r["rel_path"]: (r["sha1"], json.loads(r["chunk_ids"]))
                     for r in conn.execute("SELECT rel_path, sha1, chunk_ids FROM files")}
            seen: set[str] = set()
            for path, rel in iter_notes(vault, cfg.paths.exclude):
                stats.files_seen += 1
                seen.add(rel)
                text = path.read_text(encoding="utf-8", errors="replace")
                sha = hashlib.sha1(text.encode("utf-8")).hexdigest()
                if rel in known and known[rel][0] == sha:
                    stats.files_unchanged += 1
                    continue
                if rel in known:
                    store.delete_chunks(conn, known[rel][1])
                chunks = chunk_note(rel, text)
                vectors = emb.encode([f"{c.title}\n{c.section_heading}\n{c.text}" for c in chunks]) if chunks else None
                store.insert_chunks(conn, chunks, vectors)
                conn.execute("INSERT OR REPLACE INTO files(rel_path, sha1, chunk_ids) VALUES (?, ?, ?)",
                             (rel, sha, json.dumps([c.chunk_id for c in chunks])))
                stats.files_indexed += 1
                stats.chunks_written += len(chunks)
            for rel, (_, ids) in known.items():
                if rel not in seen:
                    store.delete_chunks(conn, ids)
                    conn.execute("DELETE FROM files WHERE rel_path = ?", (rel,))
                    stats.files_deleted += 1
            store.set_meta(conn, "schema_version", store.SCHEMA_VERSION)
            store.set_meta(conn, "embedder", emb.name)
            store.set_meta(conn, "built_at", str(int(time.time())))
            conn.commit()
            stats.dense_rows = store.export_dense(conn, cfg.dense_path, cfg.dense_ids_path, emb.name)
        finally:
            conn.close()
    stats.seconds = time.perf_counter() - t0
    log.info("index: %s", stats)
    return stats
