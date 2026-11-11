from __future__ import annotations

from pathlib import Path

import pytest

from hybrid_rag.config import load_config
from hybrid_rag.embedders import HashingEmbedder
from hybrid_rag.indexer import build_index

NOTE = """---
id: {id}
type: {type}
status: {status}
tier: {tier}
reliability: {rel}
generated_by: {gen}
verified: {verified}
volatility: {vol}
---

# {title}

## Body

{body}
"""


def write_note(vault: Path, rel: str, *, id: str, body: str, title: str = "", type: str = "reference",
               status: str = "active", tier: int = 1, rel_: float = 0.9, gen: str = "human",
               verified: str = "2026-10-01", vol: str = "slow") -> Path:
    path = vault / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(NOTE.format(id=id, type=type, status=status, tier=tier, rel=rel_, gen=gen,
                                verified=verified, vol=vol, title=title or id, body=body), encoding="utf-8")
    return path


@pytest.fixture
def vault(tmp_path: Path) -> Path:
    v = tmp_path / "vault"
    write_note(v, "a/backup-old.md", id="backup-old", status="superseded", verified="2025-01-10",
               body="Nightly backups are kept for 30 days and deleted by the lifecycle rule afterwards.")
    write_note(v, "a/backup.md", id="backup", verified="2026-09-01",
               body="Nightly backups are kept for 90 days. Monthly snapshots stay one year in cold storage.")
    write_note(v, "b/price-2025.md", id="price-2025", verified="2025-10-01", vol="fast",
               body="The Team plan costs 12 dollars per seat per month, billed monthly.")
    write_note(v, "b/price-2026.md", id="price-2026", verified="2026-10-01", vol="fast",
               body="The Team plan costs 15 dollars per seat per month, billed monthly.")
    write_note(v, "c/limits.md", id="limits", tier=0, rel_=0.95,
               body="Each API token may send 600 requests per minute with a burst allowance of 50.")
    write_note(v, "c/chat.md", id="chat", type="session", tier=3, rel_=0.5, gen="model:draft", vol="fast",
               body="In the chat we said the API token limit is maybe 100 requests per minute.")
    write_note(v, "d/adr-007.md", id="adr-007", title="ADR-007: Use blue-green deploys",
               body="Deploys switch traffic between two identical environments so a rollback takes a minute.")
    write_note(v, "d/garden.md", id="garden", body="The office garden has tomatoes, basil and two lemon trees.")
    (v / "README.md").write_text("# not a note\n\nThis readme must not be indexed.\n", encoding="utf-8")
    return v


@pytest.fixture
def cfg(tmp_path: Path, vault: Path):
    return load_config(None, env={}, paths__vault=str(vault), paths__data_dir=str(tmp_path / "idx"),
                       embedder__backend="hashing", provenance__reference_date="2026-11-01")


@pytest.fixture
def embedder():
    return HashingEmbedder(256)


@pytest.fixture
def built(cfg, embedder):
    build_index(cfg, embedder=embedder)
    return cfg
