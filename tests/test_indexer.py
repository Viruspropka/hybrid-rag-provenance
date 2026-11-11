import json
import sqlite3

import pytest

from hybrid_rag.embedders import HashingEmbedder
from hybrid_rag.indexer import _Lock, build_index
from tests.conftest import write_note


def _notes(cfg):
    with sqlite3.connect(cfg.index_path) as c:
        return sorted(r[0] for r in c.execute("SELECT DISTINCT note_id FROM chunks"))


def test_full_build(cfg, embedder):
    s = build_index(cfg, embedder=embedder)
    assert s.files_seen == 8 and s.files_indexed == 8 and s.chunks_written == s.dense_rows == 8
    assert "README" not in _notes(cfg)
    ids = json.loads(cfg.dense_ids_path.read_text())
    assert ids["embedder"] == "hashing-256" and len(ids["ids"]) == 8


def test_incremental_update_and_delete(cfg, embedder, vault):
    build_index(cfg, embedder=embedder)
    s = build_index(cfg, embedder=embedder)
    assert s.files_indexed == 0 and s.files_unchanged == 8
    write_note(vault, "d/garden.md", id="garden", body="The garden now grows peppers and chillies only.")
    (vault / "c" / "chat.md").unlink()
    s = build_index(cfg, embedder=embedder)
    assert s.files_indexed == 1 and s.files_deleted == 1 and s.dense_rows == 7
    assert "chat" not in _notes(cfg)
    with sqlite3.connect(cfg.index_path) as c:
        text = c.execute("SELECT text FROM chunks WHERE note_id='garden'").fetchone()[0]
        fts = c.execute("SELECT count(*) FROM chunks_fts WHERE chunks_fts MATCH 'tomatoes'").fetchone()[0]
    assert "peppers" in text and fts == 0


def test_embedder_change_reembeds_everything(cfg, embedder):
    build_index(cfg, embedder=embedder)
    s = build_index(cfg, embedder=HashingEmbedder(64))
    assert s.files_indexed == 8 and s.embedder == "hashing-64"
    assert json.loads(cfg.dense_ids_path.read_text())["embedder"] == "hashing-64"


def test_rebuild_and_missing_vault(cfg, embedder, tmp_path):
    build_index(cfg, embedder=embedder)
    assert build_index(cfg, rebuild=True, embedder=embedder).files_indexed == 8
    cfg.paths.vault = tmp_path / "missing"
    with pytest.raises(FileNotFoundError):
        build_index(cfg, embedder=embedder)


def test_second_writer_is_refused(cfg, embedder):
    with _Lock(cfg.paths.data_dir / "index.lock"):
        with pytest.raises(RuntimeError, match="lock"):
            build_index(cfg, embedder=embedder)


def test_hidden_and_conflict_files_are_skipped(cfg, embedder, vault):
    write_note(vault, ".obsidian/x.md", id="hidden", body="should never be indexed at all, it is config")
    write_note(vault, "_removed_backups/old.md", id="removed", body="a retired copy that must stay out")
    write_note(vault, "a/backup.sync-conflict-1.md", id="conflict", body="sync debris that must stay out")
    build_index(cfg, embedder=embedder)
    assert not {"hidden", "removed", "conflict"} & set(_notes(cfg))


def test_index_folder_owned_by_someone_else_is_refused(cfg, monkeypatch):
    import os
    from hybrid_rag.store import ensure_data_dir
    ensure_data_dir(cfg.paths.data_dir)
    assert (cfg.paths.data_dir.stat().st_mode & 0o777) == 0o700
    real = os.getuid()
    monkeypatch.setattr(os, "getuid", lambda: real + 1)
    with pytest.raises(RuntimeError, match="another user"):
        build_index(cfg)
