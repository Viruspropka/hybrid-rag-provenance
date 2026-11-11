"""Markdown notes: frontmatter parsing and heading-aligned chunking.

The frontmatter parser is deliberately tiny (no PyYAML): flat ``key: value``
pairs and inline lists ``[a, b]``. Unknown keys are ignored and a malformed
value falls back to the default, because one bad note must never stop a full
re-index.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone

_FRONTMATTER_RE = re.compile(r"\A---\r?\n(.*?)\r?\n---\r?\n?", re.DOTALL)
_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*$", re.MULTILINE)

MAX_CHUNK_CHARS = 4000      # ~1k tokens: well inside bge-m3's window
MIN_CHUNK_CHARS = 120
CHUNK_OVERLAP_CHARS = 400


@dataclass
class Frontmatter:
    """The provenance-relevant subset of a note's frontmatter."""
    id: str = ""
    title: str = ""
    doc_type: str = ""
    status: str = "active"
    tags: list[str] = field(default_factory=list)
    sensitive: bool = False          # true => the note is never indexed
    tier: int | None = None          # 0 = primary source .. 4 = synthesis of synthesis
    reliability: float | None = None # (0, 1]
    generated_by: str = ""           # human | model:<name> | tool:<name> | unknown
    verified: str = ""               # YYYY-MM-DD: last date the facts were confirmed
    volatility: str = ""             # static | slow | fast


def parse_frontmatter(text: str) -> tuple[Frontmatter, str]:
    """Split ``text`` into (frontmatter, body). No frontmatter => defaults, full text."""
    m = _FRONTMATTER_RE.match(text)
    if not m:
        return Frontmatter(), text
    fm = Frontmatter()
    for line in m.group(1).splitlines():
        if ":" not in line or line.lstrip().startswith("#"):
            continue
        key, _, value = line.partition(":")
        key = key.strip().lower()
        value = value.strip().strip("'\"")
        if not value:
            continue
        if key == "id":
            fm.id = value
        elif key == "title":
            fm.title = value
        elif key == "type":
            fm.doc_type = value.lower()
        elif key == "status":
            fm.status = value.lower()
        elif key == "sensitive":
            fm.sensitive = value.lower() in {"true", "yes", "1"}
        elif key == "tags" and value.startswith("[") and value.endswith("]"):
            fm.tags = [t.strip().strip("'\"") for t in value[1:-1].split(",") if t.strip()]
        elif key == "tier":
            try:
                fm.tier = int(value)
            except ValueError:
                pass
        elif key == "reliability":
            try:
                r = float(value)
                if 0.0 < r <= 1.0:
                    fm.reliability = r
            except ValueError:
                pass
        elif key == "generated_by":
            fm.generated_by = value.lower()
        elif key == "verified":
            fm.verified = value
        elif key == "volatility":
            fm.volatility = value.lower()
    return fm, text[m.end():]


def verified_epoch(verified: str) -> float | None:
    """``YYYY-MM-DD`` -> epoch seconds (UTC midnight); missing or bad => ``None``."""
    if not verified:
        return None
    try:
        d = datetime.strptime(verified[:10], "%Y-%m-%d")
    except ValueError:
        return None
    return d.replace(tzinfo=timezone.utc).timestamp()


@dataclass
class Chunk:
    """One indexable unit with the provenance of the note it came from."""
    chunk_id: str
    note_id: str
    file_path: str
    title: str
    section_heading: str
    text: str
    status: str
    doc_type: str
    tags: list[str]
    tier: int | None
    reliability: float | None
    generated_by: str
    verified: float | None
    volatility: str


def _heading_at(body: str, offset: int) -> str:
    heading = ""
    for m in _HEADING_RE.finditer(body):
        if m.start() > offset:
            break
        heading = m.group(2).strip()
    return heading


def _split_sections(body: str) -> list[str]:
    """Split on level-2 headings, keeping the ``## `` with its section."""
    sections: list[str] = []
    pos = 0
    while True:
        idx = body.find("\n## ", pos)
        if idx == -1:
            sections.append(body[pos:])
            return sections
        sections.append(body[pos:idx])
        pos = idx + 1


def chunk_note(rel_path: str, text: str, *, max_chars: int = MAX_CHUNK_CHARS,
               min_chars: int = MIN_CHUNK_CHARS, overlap: int = CHUNK_OVERLAP_CHARS) -> list[Chunk]:
    """Chunk one note: one chunk per ``##`` section, long sections windowed with overlap.

    Pieces shorter than ``min_chars`` are merged into their neighbour instead of
    being dropped, so a note made of short sections still has all its text indexed.
    """
    fm, body = parse_frontmatter(text)
    if fm.sensitive or not body.strip():
        return []
    stem = rel_path.rsplit("/", 1)[-1]
    note_id = fm.id or (stem[:-3] if stem.endswith(".md") else stem)
    title_m = _HEADING_RE.search(body)
    title = fm.title or (title_m.group(2).strip() if title_m and title_m.group(1) == "#" else note_id)
    pieces: list[str] = []
    for section in _split_sections(body):
        section = section.strip()
        if not section:
            continue
        if len(section) <= max_chars:
            pieces.append(section)
            continue
        step = max(1, max_chars - overlap)
        for i in range(0, len(section), step):
            piece = section[i:i + max_chars]
            if piece.strip():
                pieces.append(piece)
            if i + max_chars >= len(section):
                break
    # Short pieces (a bare title line, a two-line section) are merged into the next
    # piece, or into the previous one at the end, so no text is ever dropped.
    kept: list[tuple[str, str]] = []          # (text, heading)
    carry = ""
    for piece in pieces:
        heading = _heading_at(body, max(body.find(piece[:80]), 0)) or title
        text_ = f"{carry}\n\n{piece}" if carry else piece
        if len(text_) < min_chars:
            carry = text_
            continue
        kept.append((text_, heading))
        carry = ""
    if carry:
        if kept:
            kept[-1] = (f"{kept[-1][0]}\n\n{carry}", kept[-1][1])
        else:
            kept.append((carry, _heading_at(body, len(body)) or title))
    chunks: list[Chunk] = []
    for n, (piece, heading) in enumerate(kept):
        digest = hashlib.sha1(f"{rel_path}|{n}|{heading}|{piece[:120]}".encode("utf-8")).hexdigest()[:16]
        chunks.append(Chunk(
            chunk_id=digest,
            note_id=note_id,
            file_path=rel_path,
            title=title,
            section_heading=heading,
            text=piece.strip(),
            status=fm.status or "active",
            doc_type=fm.doc_type or "note",
            tags=list(fm.tags),
            tier=fm.tier,
            reliability=fm.reliability,
            generated_by=fm.generated_by or "unknown",
            verified=verified_epoch(fm.verified),
            volatility=fm.volatility,
        ))
    return chunks
