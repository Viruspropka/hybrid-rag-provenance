"""Embedders behind one small interface.

* ``HashingEmbedder``: deterministic feature hashing of words, word bigrams and
  character trigrams. No download, no randomness, runs anywhere. It is a mock
  for tests and offline demos: its "semantics" are lexical, so it shows how the
  pipeline behaves, not how good a neural model is.
* ``SentenceTransformerEmbedder``: any sentence-transformers model, ``BAAI/bge-m3``
  by default. Needs the optional ``models`` extra; weights are downloaded by the
  user on first use and are never part of this repository.

``make_embedder`` tries the configured backend and falls back (logged) to the
configured fallback, so a machine without the heavy extras still works.
"""

from __future__ import annotations

import logging
import zlib
from typing import Protocol

import numpy as np

from .config import EmbedderConfig
from .text import tokens

log = logging.getLogger(__name__)


class Embedder(Protocol):
    """Anything that turns texts into L2-normalised float32 rows."""
    name: str   # identity stored in the index; a query must use the same embedder
    dim: int

    def encode(self, texts: list[str]) -> np.ndarray: ...


def _normalise(mat: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return (mat / norms).astype(np.float32)


class HashingEmbedder:
    """Signed feature hashing (crc32, stable across processes) with sublinear tf."""

    def __init__(self, dim: int = 512) -> None:
        if dim < 16:
            raise ValueError("hashing dim must be >= 16")
        self.dim = dim
        self.name = f"hashing-{dim}"

    def _features(self, text: str) -> list[tuple[str, float]]:
        words = tokens(text)
        feats: list[tuple[str, float]] = [(f"w:{w}", 1.0) for w in words]
        feats += [(f"b:{a}_{b}", 0.5) for a, b in zip(words, words[1:])]
        for w in words:
            padded = f"<{w}>"
            feats += [(f"c:{padded[i:i + 3]}", 0.25) for i in range(len(padded) - 2)]
        return feats

    def encode(self, texts: list[str]) -> np.ndarray:
        mat = np.zeros((len(texts), self.dim), dtype=np.float64)
        for row, text in enumerate(texts):
            counts: dict[int, float] = {}
            for feat, weight in self._features(text):
                h = zlib.crc32(feat.encode("utf-8"))
                idx = h % self.dim
                sign = 1.0 if (h >> 31) & 1 == 0 else -1.0
                counts[idx] = counts.get(idx, 0.0) + sign * weight
            for idx, value in counts.items():
                mat[row, idx] = np.sign(value) * np.log1p(abs(value))
        return _normalise(mat)


class SentenceTransformerEmbedder:
    """Lazy wrapper around ``sentence_transformers.SentenceTransformer``."""

    def __init__(self, model: str, device: str = "auto", max_seq_length: int = 1536,
                 batch_size: int = 16) -> None:
        from sentence_transformers import SentenceTransformer  # optional extra

        if device == "auto":
            device = _pick_device()
        log.info("loading embedder %s on %s", model, device)
        self._model = SentenceTransformer(model, device=device)
        self._model.max_seq_length = max_seq_length
        self._batch = batch_size
        get_dim = getattr(self._model, "get_embedding_dimension", None) or self._model.get_sentence_embedding_dimension
        self.dim = int(get_dim())
        self.name = f"st:{model}"

    def encode(self, texts: list[str]) -> np.ndarray:
        vecs = self._model.encode(texts, batch_size=self._batch, normalize_embeddings=True,
                                  convert_to_numpy=True, show_progress_bar=False)
        return np.asarray(vecs, dtype=np.float32)


def _pick_device() -> str:
    try:
        import torch
    except ImportError:
        return "cpu"
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _build(backend: str, cfg: EmbedderConfig) -> Embedder:
    backend = backend.lower()
    if backend in ("hashing", "mock"):
        return HashingEmbedder(cfg.hashing_dim)
    if backend in ("sentence-transformers", "st"):
        return SentenceTransformerEmbedder(cfg.model, cfg.device, cfg.max_seq_length, cfg.batch_size)
    raise ValueError(f"unknown embedder backend '{backend}'")


def make_embedder(cfg: EmbedderConfig) -> Embedder:
    """The configured embedder, or the fallback when it cannot be loaded."""
    try:
        return _build(cfg.backend, cfg)
    except (ImportError, OSError, RuntimeError, ValueError) as exc:
        if not cfg.fallback or cfg.fallback == cfg.backend:
            raise
        log.warning("embedder '%s' (%s) unavailable: %s; using fallback '%s'",
                    cfg.backend, cfg.model, exc, cfg.fallback)
        return _build(cfg.fallback, cfg)
