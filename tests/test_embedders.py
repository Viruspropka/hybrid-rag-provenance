import builtins

import numpy as np
import pytest

from hybrid_rag.config import EmbedderConfig
from hybrid_rag.embedders import HashingEmbedder, make_embedder


def test_hashing_is_deterministic_and_normalised():
    e = HashingEmbedder(128)
    a = e.encode(["backup retention policy", ""])
    b = HashingEmbedder(128).encode(["backup retention policy", ""])
    assert np.array_equal(a, b)
    assert a.dtype == np.float32 and a.shape == (2, 128)
    assert np.linalg.norm(a[0]) == pytest.approx(1.0, abs=1e-5)
    assert np.linalg.norm(a[1]) == 0.0


def test_hashing_similarity_is_lexical():
    e = HashingEmbedder(512)
    q, near, far = e.encode(["how long are backups kept", "backups are kept for 90 days", "tomatoes and basil"])
    assert q @ near > q @ far


def test_fallback_when_sentence_transformers_missing(monkeypatch, caplog):
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name.startswith("sentence_transformers"):
            raise ImportError("not installed")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    emb = make_embedder(EmbedderConfig(backend="sentence-transformers", hashing_dim=64))
    assert emb.name == "hashing-64"
    assert "using fallback" in caplog.text


def test_no_fallback_raises():
    with pytest.raises(ValueError):
        make_embedder(EmbedderConfig(backend="nope", fallback=""))
    with pytest.raises(ValueError):
        HashingEmbedder(4)
