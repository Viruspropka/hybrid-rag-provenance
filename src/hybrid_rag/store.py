"""Index storage: one SQLite file (chunks, FTS5, vectors) plus an exported dense matrix.

* ``chunks``      chunk text and the note's provenance fields
* ``chunks_fts``  FTS5 (porter + unicode61) over the chunk text, BM25 ranking
* ``vectors``     one float32 blob per chunk, the source of truth for dense search
* ``files``       sha1 per note, for incremental re-indexing
* ``meta``        embedder identity and build time

At the end of every index run the vectors are exported as ``dense.npy`` (an
``N x dim`` matrix, memory-mapped at query time) with ``dense_ids.json``. If that
pair is missing, or was built by another embedder, the dense index counts as
unavailable and retrieval degrades to keyword search.
"""

from __future__ import annotations

import json
import os
import sqlite3
import threading
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .notes import Chunk

SCHEMA_VERSION = "1"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS files (
    rel_path TEXT PRIMARY KEY, sha1 TEXT NOT NULL, chunk_ids TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS chunks (
    chunk_id TEXT PRIMARY KEY,
    note_id TEXT NOT NULL,
    file_path TEXT NOT NULL,
    title TEXT NOT NULL,
    section_heading TEXT NOT NULL,
    text TEXT NOT NULL,
    status TEXT NOT NULL,
    doc_type TEXT NOT NULL,
    tags TEXT NOT NULL,
    tier INTEGER,
    reliability REAL,
    generated_by TEXT NOT NULL,
    verified REAL,
    volatility TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS chunks_note ON chunks(note_id);
CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
    text, title, chunk_id UNINDEXED, tokenize = 'porter unicode61'
);
CREATE TABLE IF NOT EXISTS vectors (chunk_id TEXT PRIMARY KEY, vec BLOB NOT NULL);
"""

CHUNK_COLUMNS = ("chunk_id", "note_id", "file_path", "title", "section_heading", "text", "status",
                 "doc_type", "tags", "tier", "reliability", "generated_by", "verified", "volatility")


def ensure_data_dir(folder: Path) -> None:
    """Create the index folder 0700, or check an existing one belongs to this user.

    Every index file lives in this folder, so once it is ours and not world-writable
    nobody else can plant a symlink or a fake index inside it.
    """
    if not folder.exists():
        folder.mkdir(mode=0o700, parents=True)
        return
    check_data_dir(folder)


def check_data_dir(folder: Path) -> None:
    """Raise unless an existing index folder belongs to this user and only they can write it."""
    st = folder.stat()
    if hasattr(os, "getuid") and st.st_uid != os.getuid():
        raise RuntimeError(f"index folder {folder} belongs to another user; refusing to use it")
    if st.st_mode & 0o022:
        raise RuntimeError(f"index folder {folder} is writable by other users; refusing to use it")


def connect(path: Path) -> sqlite3.Connection:
    ensure_data_dir(path.parent)
    conn = sqlite3.connect(path, timeout=30.0)
    conn.row_factory = sqlite3.Row
    try:
        conn.execute("PRAGMA journal_mode=WAL")   # readers keep working during a re-index
    except sqlite3.OperationalError:
        pass
    conn.execute("PRAGMA busy_timeout=30000")
    conn.executescript(_SCHEMA)
    return conn


def get_meta(conn: sqlite3.Connection, key: str) -> str | None:
    row = conn.execute("SELECT value FROM meta WHERE key = ?", (key,)).fetchone()
    return row[0] if row else None


def set_meta(conn: sqlite3.Connection, key: str, value: str) -> None:
    conn.execute("INSERT OR REPLACE INTO meta(key, value) VALUES (?, ?)", (key, value))


def insert_chunks(conn: sqlite3.Connection, chunks: list[Chunk], vectors: np.ndarray | None) -> None:
    for i, c in enumerate(chunks):
        conn.execute(
            f"INSERT OR REPLACE INTO chunks({','.join(CHUNK_COLUMNS)}) VALUES ({','.join('?' * len(CHUNK_COLUMNS))})",
            (c.chunk_id, c.note_id, c.file_path, c.title, c.section_heading, c.text, c.status, c.doc_type,
             json.dumps(c.tags), c.tier, c.reliability, c.generated_by, c.verified, c.volatility or ""),
        )
        conn.execute("DELETE FROM chunks_fts WHERE chunk_id = ?", (c.chunk_id,))
        conn.execute("INSERT INTO chunks_fts(text, title, chunk_id) VALUES (?, ?, ?)",
                     (c.text, f"{c.title} {c.section_heading}", c.chunk_id))
        if vectors is not None:
            conn.execute("INSERT OR REPLACE INTO vectors(chunk_id, vec) VALUES (?, ?)",
                         (c.chunk_id, np.asarray(vectors[i], dtype=np.float32).tobytes()))


def delete_chunks(conn: sqlite3.Connection, chunk_ids: list[str]) -> None:
    for cid in chunk_ids:
        conn.execute("DELETE FROM chunks WHERE chunk_id = ?", (cid,))
        conn.execute("DELETE FROM chunks_fts WHERE chunk_id = ?", (cid,))
        conn.execute("DELETE FROM vectors WHERE chunk_id = ?", (cid,))


def export_dense(conn: sqlite3.Connection, matrix_path: Path, ids_path: Path, embedder_name: str) -> int:
    """Write the dense matrix + id list atomically; returns the row count."""
    rows = conn.execute("SELECT chunk_id, vec FROM vectors ORDER BY chunk_id").fetchall()
    ids = [r["chunk_id"] for r in rows]
    if rows:
        mat = np.vstack([np.frombuffer(r["vec"], dtype=np.float32) for r in rows])
    else:
        mat = np.zeros((0, 1), dtype=np.float32)
    tmp_m = matrix_path.with_suffix(".tmp.npy")
    np.save(tmp_m, mat)
    tmp_i = ids_path.with_suffix(".tmp")
    tmp_i.write_text(json.dumps({"embedder": embedder_name, "ids": ids}), encoding="utf-8")
    tmp_m.replace(matrix_path)
    tmp_i.replace(ids_path)
    return len(ids)


@dataclass
class DenseIndex:
    """Brute-force cosine search over a memory-mapped matrix.

    Fine up to a few hundred thousand chunks on a laptop. For larger corpora put
    an ANN index (FAISS, LanceDB, ...) behind the same ``search`` signature.
    """
    ids: list[str]
    matrix: np.ndarray
    embedder: str

    @classmethod
    def load(cls, matrix_path: Path, ids_path: Path) -> "DenseIndex":
        if not matrix_path.is_file() or not ids_path.is_file():
            raise FileNotFoundError(f"dense index missing ({matrix_path.name})")
        if ids_path.parent.exists():
            check_data_dir(ids_path.parent)
        meta = json.loads(ids_path.read_text(encoding="utf-8"))
        matrix = np.load(matrix_path, mmap_mode="r", allow_pickle=False)
        if matrix.shape[0] != len(meta["ids"]):
            raise ValueError("dense matrix and id list disagree; re-index")
        return cls(ids=list(meta["ids"]), matrix=matrix, embedder=meta["embedder"])

    def search(self, qvec: np.ndarray, top_k: int) -> list[tuple[str, float]]:
        if not self.ids:
            return []
        scores = np.asarray(self.matrix @ np.asarray(qvec, dtype=np.float32))
        top_k = min(top_k, len(self.ids))
        idx = np.argpartition(-scores, top_k - 1)[:top_k]
        # stable tie-break on chunk id keeps runs reproducible
        order = sorted(idx.tolist(), key=lambda i: (-float(scores[i]), self.ids[i]))
        return [(self.ids[i], float(scores[i])) for i in order]


class ThreadLocalConnections:
    """One SQLite connection per thread (SQLite forbids sharing them across threads)."""

    def __init__(self, path: Path) -> None:
        self._path = path
        self._local = threading.local()

    def get(self) -> sqlite3.Connection:
        conn = getattr(self._local, "conn", None)
        if conn is None:
            if self._path.parent.exists():
                check_data_dir(self._path.parent)
            if not self._path.is_file():
                raise FileNotFoundError(f"index missing at {self._path}; run `hrag index` first")
            conn = sqlite3.connect(self._path, timeout=30.0)
            conn.row_factory = sqlite3.Row
            self._local.conn = conn
        return conn
