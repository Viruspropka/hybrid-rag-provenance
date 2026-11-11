"""Configuration: a TOML file plus environment variables, no hardcoded paths.

Lookup order (later wins):

1. Built-in defaults (the dataclasses below).
2. The TOML file named by ``HRAG_CONFIG``, else ``./hrag.toml`` if it exists.
3. Environment variables (``HRAG_VAULT``, ``HRAG_DATA_DIR``, ...), see ``ENV_OVERRIDES``.
4. Explicit keyword overrides from the CLI.

Relative paths in the TOML file are resolved against the file's own directory,
relative paths from the environment against the current working directory.
"""

from __future__ import annotations

import copy
import logging
import os
import tomllib
from dataclasses import dataclass, field, fields, is_dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)


SAMPLE_VAULT = Path(__file__).resolve().parent / "sample_vault"


@dataclass
class PathsConfig:
    vault: Path = SAMPLE_VAULT          # the bundled synthetic vault; point it at your notes
    data_dir: Path = Path(".hrag")
    exclude: list[str] = field(default_factory=lambda: ["README.md"])   # fnmatch patterns on the vault-relative path ("*" also crosses "/")


@dataclass
class EmbedderConfig:
    # "sentence-transformers" loads ``model`` (bge-m3 by default) when the optional
    # ``models`` extra is installed; otherwise ``fallback`` is used and logged.
    backend: str = "sentence-transformers"
    model: str = "BAAI/bge-m3"
    fallback: str = "hashing"
    device: str = "auto"            # auto | cpu | mps | cuda
    max_seq_length: int = 1536      # chunks are <= ~1k tokens; keeps attention memory bounded
    batch_size: int = 16
    hashing_dim: int = 512


@dataclass
class RerankerConfig:
    backend: str = "none"           # none | overlap | cross-encoder | flashrank
    model: str = "BAAI/bge-reranker-v2-m3"
    flashrank_model: str = "ms-marco-MultiBERT-L-12"
    flashrank_cache: str = ""       # "" = flashrank's own default cache dir
    min_spread: float = 0.01        # a flatter score band means "no opinion": keep RRF order
    pool: int = 24                  # candidates sent to the reranker


@dataclass
class RetrievalConfig:
    dense_top_k: int = 20
    sparse_top_k: int = 20
    rrf_k: int = 20                 # the RRF paper uses 60; see README "Choosing rrf_k"
    max_chunks_per_note: int = 2
    inference_timeout_s: float = 25.0
    sparse_overfetch: int = 4
    low_confidence_coverage: float = 0.34   # top hit covering fewer query terms => low confidence
    historical_patterns: list[str] = field(default_factory=lambda: [
        r"\b(history|historical|previously|used to|originally|at the time)\b",
        r"\b(superseded|archived|deprecated|old version)\b",
    ])
    mem_min_available_pct: float = 10.0
    mem_max_swap_used_pct: float = 90.0


@dataclass
class ProvenanceConfig:
    enabled: bool = True            # master switch: off => every multiplier is 1.0
    status_filter: bool = True      # hide hidden_statuses unless the query is historical
    tier: bool = True
    trust: bool = True
    freshness: bool = True
    status: bool = True
    doc_type: bool = True
    reference_date: str = ""        # "" = today (UTC); set YYYY-MM-DD for reproducible runs
    default_reliability: float = 0.85
    default_volatility: str = "slow"
    trust_floor: float = 0.5
    freshness_floor: float = 0.8
    half_life_days: dict[str, float] = field(default_factory=lambda: {"slow": 365.0, "fast": 45.0})
    hidden_statuses: list[str] = field(default_factory=lambda: ["superseded", "archived"])
    status_multipliers: dict[str, float] = field(default_factory=lambda: {
        "active": 1.0, "draft": 0.9, "superseded": 0.5, "archived": 0.3,
    })
    tier_multipliers: dict[str, float] = field(default_factory=lambda: {
        "0": 1.05, "1": 1.02, "2": 1.00, "3": 0.97, "4": 0.95,
    })
    authorship_weights: dict[str, float] = field(default_factory=lambda: {
        "human": 1.0, "tool": 0.98, "model": 0.95, "unknown": 0.95,
    })
    # Only ephemera are demoted; canon types stay at 1.0. Boosting canon types as well
    # (runbook 1.10, ...) let weakly relevant runbooks overtake the right note.
    type_multipliers: dict[str, float] = field(default_factory=lambda: {
        "session": 0.60, "research": 0.85,
    })


@dataclass
class FastPathConfig:
    # Exact-key fast path: a query naming a key such as ``ADR-012`` or ``INC-042``
    # gets the note that carries that key in its id/title/heading first (score 1.0).
    # Empty pattern = plugin off.
    exact_key_pattern: str = r"\b(?:ADR|INC|RB)-\d{1,4}\b"


@dataclass
class DaemonConfig:
    socket: str = ""                # "" => $TMPDIR/hrag-<uid>.sock
    admit_wait_s: float = 5.0       # wait this long for the heavy slot, then degrade
    admit_max_waiters: int = 4
    max_query_chars: int = 2000


@dataclass
class Config:
    paths: PathsConfig = field(default_factory=PathsConfig)
    embedder: EmbedderConfig = field(default_factory=EmbedderConfig)
    reranker: RerankerConfig = field(default_factory=RerankerConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    provenance: ProvenanceConfig = field(default_factory=ProvenanceConfig)
    fastpath: FastPathConfig = field(default_factory=FastPathConfig)
    daemon: DaemonConfig = field(default_factory=DaemonConfig)

    # ------------------------------------------------------------------ helpers
    def copy(self) -> "Config":
        return copy.deepcopy(self)

    @property
    def index_path(self) -> Path:
        return self.paths.data_dir / "index.sqlite"

    @property
    def dense_path(self) -> Path:
        return self.paths.data_dir / "dense.npy"

    @property
    def dense_ids_path(self) -> Path:
        return self.paths.data_dir / "dense_ids.json"

    def socket_path(self) -> str:
        if self.daemon.socket:
            return self.daemon.socket
        # A per-user folder (created 0700 by the daemon): /tmp is shared on Linux.
        tmp = os.environ.get("TMPDIR", "/tmp").rstrip("/")
        uid = os.getuid() if hasattr(os, "getuid") else 0
        return f"{tmp}/hrag-{uid}/hrag.sock"

    def reference_epoch(self) -> float:
        """The "now" used by the freshness clock (fixed when reference_date is set)."""
        if self.provenance.reference_date:
            d = date.fromisoformat(self.provenance.reference_date[:10])
            return datetime(d.year, d.month, d.day, tzinfo=timezone.utc).timestamp()
        return datetime.now(tz=timezone.utc).timestamp()


# env var -> (section, key, type)
ENV_OVERRIDES: dict[str, tuple[str, str, type]] = {
    "HRAG_VAULT": ("paths", "vault", Path),
    "HRAG_DATA_DIR": ("paths", "data_dir", Path),
    "HRAG_EMBEDDER": ("embedder", "backend", str),
    "HRAG_EMBED_MODEL": ("embedder", "model", str),
    "HRAG_EMBED_DEVICE": ("embedder", "device", str),
    "HRAG_RERANKER": ("reranker", "backend", str),
    "HRAG_RERANKER_MODEL": ("reranker", "model", str),
    "HRAG_PROVENANCE": ("provenance", "enabled", bool),
    "HRAG_REFERENCE_DATE": ("provenance", "reference_date", str),
    "HRAG_SOCKET": ("daemon", "socket", str),
}


def _coerce(value: Any, typ: type) -> Any:
    if typ is bool:
        if isinstance(value, bool):
            return value
        return str(value).strip().lower() not in ("", "0", "false", "no", "off")
    if typ is Path:
        return Path(str(value)).expanduser()
    return typ(value)


def _apply_section(target: Any, values: dict[str, Any], base_dir: Path | None) -> None:
    known = {f.name: f for f in fields(target)}
    for key, value in values.items():
        if key not in known:
            raise ValueError(f"unknown config key '{key}' in section {type(target).__name__}")
        current = getattr(target, key)
        if isinstance(current, Path):
            p = Path(str(value)).expanduser()
            if base_dir is not None and not p.is_absolute():
                p = (base_dir / p).resolve()
            setattr(target, key, p)
        elif isinstance(current, dict):
            merged = dict(current)
            merged.update({str(k): v for k, v in value.items()})
            setattr(target, key, merged)
        elif isinstance(current, bool):
            setattr(target, key, _coerce(value, bool))
        elif isinstance(current, (int, float)) and not isinstance(current, bool):
            setattr(target, key, type(current)(value))
        else:
            setattr(target, key, value)


def load_config(path: str | Path | None = None, *, env: dict[str, str] | None = None,
                **overrides: Any) -> Config:
    """Build a ``Config`` from defaults, an optional TOML file, env vars and overrides.

    ``overrides`` use ``section__key`` names, e.g. ``embedder__backend="hashing"``.
    """
    env = os.environ if env is None else env
    cfg = Config()
    file_path: Path | None = None
    if path is not None:
        file_path = Path(path).expanduser()
    elif env.get("HRAG_CONFIG"):
        file_path = Path(env["HRAG_CONFIG"]).expanduser()
    elif Path("hrag.toml").is_file():
        # Only a config the user owns: a hrag.toml dropped into a shared folder by
        # someone else must not pick the model or the data folder.
        if hasattr(os, "getuid") and Path("hrag.toml").stat().st_uid != os.getuid():
            log.warning("ignoring ./hrag.toml: it belongs to another user")
        else:
            file_path = Path("hrag.toml")
    if file_path is not None:
        if not file_path.is_file():
            raise FileNotFoundError(f"config file not found: {file_path}")
        data = tomllib.loads(file_path.read_text(encoding="utf-8"))
        base = file_path.resolve().parent
        for section, values in data.items():
            if not hasattr(cfg, section) or not is_dataclass(getattr(cfg, section)):
                raise ValueError(f"unknown config section [{section}]")
            _apply_section(getattr(cfg, section), values, base)
    for var, (section, key, typ) in ENV_OVERRIDES.items():
        if env.get(var) not in (None, ""):
            setattr(getattr(cfg, section), key, _coerce(env[var], typ))
    for name, value in overrides.items():
        if value is None:
            continue
        section, _, key = name.partition("__")
        target = getattr(cfg, section)
        current = getattr(target, key)
        setattr(target, key, _coerce(value, type(current)) if not isinstance(current, (dict, list)) else value)
    return cfg
