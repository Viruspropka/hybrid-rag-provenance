from pathlib import Path

import pytest

from hybrid_rag.config import SAMPLE_VAULT, load_config


def test_defaults_need_no_file(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    cfg = load_config(None, env={})
    assert cfg.paths.vault == SAMPLE_VAULT and cfg.embedder.model == "BAAI/bge-m3"
    assert cfg.retrieval.rrf_k == 20 and cfg.provenance.enabled


def test_toml_relative_paths_resolve_against_the_file(tmp_path):
    (tmp_path / "conf").mkdir()
    f = tmp_path / "conf" / "hrag.toml"
    f.write_text('[paths]\nvault = "../notes"\ndata_dir = "idx"\n[provenance]\ntrust = false\n'
                 '[provenance.half_life_days]\nfast = 30\n[retrieval]\nrrf_k = 60\n', encoding="utf-8")
    cfg = load_config(f, env={})
    assert cfg.paths.vault == (tmp_path / "notes").resolve()
    assert cfg.paths.data_dir == (tmp_path / "conf" / "idx").resolve()
    assert cfg.provenance.trust is False and cfg.retrieval.rrf_k == 60
    assert cfg.provenance.half_life_days == {"slow": 365.0, "fast": 30}


def test_env_then_overrides(tmp_path):
    env = {"HRAG_VAULT": str(tmp_path / "v"), "HRAG_EMBEDDER": "hashing", "HRAG_PROVENANCE": "0",
           "HRAG_REFERENCE_DATE": "2026-01-01"}
    cfg = load_config(None, env=env, embedder__backend="sentence-transformers")
    assert cfg.paths.vault == tmp_path / "v"
    assert cfg.provenance.enabled is False
    assert cfg.embedder.backend == "sentence-transformers"
    assert cfg.reference_epoch() == pytest.approx(1767225600.0)


def test_config_file_from_env(tmp_path):
    f = tmp_path / "x.toml"
    f.write_text('[embedder]\nbackend = "hashing"\n', encoding="utf-8")
    assert load_config(None, env={"HRAG_CONFIG": str(f)}).embedder.backend == "hashing"


@pytest.mark.parametrize("text", ['[paths]\nnope = 1\n', '[nosuch]\na = 1\n'])
def test_unknown_keys_are_errors(tmp_path, text):
    f = tmp_path / "bad.toml"
    f.write_text(text, encoding="utf-8")
    with pytest.raises(ValueError):
        load_config(f, env={})


def test_missing_file_is_an_error(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_config(tmp_path / "none.toml", env={})


def test_example_config_parses():
    example = Path(__file__).resolve().parents[1] / "hrag.example.toml"
    cfg = load_config(example, env={})
    assert cfg.embedder.model == "BAAI/bge-m3"


def test_cwd_config_owned_by_someone_else_is_ignored(tmp_path, monkeypatch):
    import os
    (tmp_path / "hrag.toml").write_text('[retrieval]\nrrf_k = 99\n', encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    assert load_config(None, env={}).retrieval.rrf_k == 99
    real = os.getuid()
    monkeypatch.setattr(os, "getuid", lambda: real + 1)
    assert load_config(None, env={}).retrieval.rrf_k == 20
