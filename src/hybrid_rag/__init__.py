"""Hybrid retrieval (dense + FTS5 + RRF) with a provenance ranking layer for markdown notes."""

__version__ = "0.1.0"

from .config import Config, load_config
from .indexer import build_index
from .retriever import Hit, RetrievalResult, Retriever

__all__ = ["Config", "Hit", "RetrievalResult", "Retriever", "build_index", "load_config", "__version__"]
