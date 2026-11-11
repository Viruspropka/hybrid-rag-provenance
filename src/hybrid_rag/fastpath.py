"""Fast-path plugins: deterministic answers that skip the fuzzy pipeline.

A fast path gets the query and a SQLite connection and returns chunk rows that
must come first. The built-in ``ExactKeyFastPath`` answers queries that name an
exact key (``ADR-012``, ``INC-042``): a key is an identifier, not a semantic
question, so it is resolved by lookup. Add your own by implementing ``FastPath``
and passing it to ``Retriever(fastpaths=[...])``.
"""

from __future__ import annotations

import re
import sqlite3
from typing import Protocol


class FastPath(Protocol):
    name: str

    def rows(self, query: str, conn: sqlite3.Connection, k: int) -> list[sqlite3.Row]: ...

    def covers_query(self, query: str) -> bool:
        """True when nothing but keys is in the query (the fuzzy path can be skipped)."""
        ...


class ExactKeyFastPath:
    name = "exact-key"

    def __init__(self, pattern: str) -> None:
        self._re = re.compile(pattern, re.IGNORECASE)

    def keys(self, query: str) -> list[str]:
        return list(dict.fromkeys(m.group(0).upper() for m in self._re.finditer(query)))

    def rows(self, query: str, conn: sqlite3.Connection, k: int) -> list[sqlite3.Row]:
        out: list[sqlite3.Row] = []
        seen: set[str] = set()
        for key in self.keys(query):
            like = f"%{key}%"
            rows = conn.execute(
                "SELECT * FROM chunks WHERE upper(note_id) = ? OR upper(title) LIKE ? "
                "OR upper(section_heading) LIKE ? ORDER BY note_id, chunk_id",
                (key, like, like),
            ).fetchall()
            # notes whose id is the key come first, then title matches, then heading matches
            rows = sorted(rows, key=lambda r: (r["note_id"].upper() != key,
                                               key not in r["title"].upper(), r["chunk_id"]))
            for r in rows:
                if r["chunk_id"] not in seen:
                    seen.add(r["chunk_id"])
                    out.append(r)
                if len(out) >= k:
                    return out
        return out

    def covers_query(self, query: str) -> bool:
        rest = self._re.sub(" ", query)
        words = [w for w in re.findall(r"[A-Za-z]{3,}", rest)
                 if w.lower() not in {"what", "show", "the", "and", "say", "says", "about", "note"}]
        return bool(self.keys(query)) and len(words) < 2


def make_fastpaths(pattern: str) -> list[FastPath]:
    return [ExactKeyFastPath(pattern)] if pattern else []
