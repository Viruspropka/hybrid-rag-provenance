# hybrid-rag-provenance

A small, self-contained retriever for a folder of markdown notes. It combines dense
embeddings and SQLite FTS5 keyword search with reciprocal-rank fusion, then re-orders the
shortlist with a **provenance layer**: each note's frontmatter (`tier`, `reliability`,
`generated_by`, `verified`, `volatility`, `status`, `type`) turns into a bounded score
multiplier. A stale price list, a chat summary written by a model, or a superseded runbook
can still be found, but it no longer outranks the current, checked source.

It runs fully offline. The core needs only Python 3.11+ and numpy. Neural models are an
optional extra, and their weights are downloaded by you on first use; none are in this repo.

The bundled sample vault (21 notes) and the 15 labelled eval queries are synthetic: a
fictional company "Quillmoor Labs" and its fictional product "Tidewell".

## Quickstart (no API keys, no model download)

```bash
python3 -m venv .venv && . .venv/bin/activate
pip install -r requirements-dev.txt     # numpy + pytest
pip install -e .                        # installs the `hrag` command
hrag demo                               # nine scenarios on the sample vault, ends with "demo OK"
python -m pytest -q                     # 75 tests
```

Without installing: `PYTHONPATH=src python -m hybrid_rag demo`.

Use it on your own notes:

```bash
hrag index --vault ~/notes --data-dir ~/.cache/hrag --embedder hashing
hrag query "how long do we keep backups?" --vault ~/notes --data-dir ~/.cache/hrag --embedder hashing --explain
hrag serve --vault ~/notes --data-dir ~/.cache/hrag --embedder hashing &   # warm daemon on a Unix socket
hrag query "how long do we keep backups?" --data-dir ~/.cache/hrag --embedder hashing   # served by the daemon
```

With a real embedding model (downloads about 2.2 GB of weights the first time):

```bash
pip install -e ".[models]"
hrag index --vault ~/notes --data-dir ~/.cache/hrag      # default model: BAAI/bge-m3
```

## Architecture in ten lines

1. `indexer` walks the vault, parses frontmatter, splits each note on `##` headings (long sections are windowed with overlap, short ones merged), and re-indexes only files whose sha1 changed.
2. Chunks and their provenance fields go into one SQLite file: a `chunks` table plus an FTS5 table (`porter unicode61`).
3. Dense vectors go into `dense.npy` (searched by brute-force cosine) with `dense_ids.json`, which also records which embedder built them.
4. A query runs BM25 on FTS5 and cosine on the dense matrix, each to depth 20.
5. The two ranked lists are fused by RRF: `sum 1/(k + rank + 1)`, `k = 20`; relevance = fused / best fused.
6. An optional reranker (cross-encoder or FlashRank) re-scores the pool; if it fails, times out or gives a flat spread, the fused order is kept.
7. The status filter drops superseded and archived notes unless the query is historical ("previously", "used to", ...).
8. The provenance layer multiplies relevance by `M = tier x trust x freshness x status x type`, then caps chunks per note and flags low term coverage.
9. If the dense side is unusable (no file, wrong embedder, model will not load, encode timeout, memory pressure) the query degrades to keyword-only and says why.
10. `hrag serve` keeps the model warm behind a Unix socket (mode 600, in a 0700 per-user folder) or loopback TCP; the client falls back to in-process keyword search when no daemon answers.

Exact-key lookups (`ADR-012`, `INC-042`, `RB-7`) go through a small plugin fast path before
any of this. The pattern is configurable, and an empty pattern turns the plugin off.

## The provenance layer

The layer runs after relevance scoring. It re-orders a relevant shortlist; it cannot pull an
irrelevant chunk in, and every factor is bounded, so a weak note is demoted, never erased.

```
final     = relevance x M
M         = tier x trust x freshness x status x type

tier      = {0: 1.05, 1: 1.02, 2: 1.00, 3: 0.97, 4: 0.95}[tier]          missing -> 1.0
trust     = 0.5 + 0.5 x min(1, reliability x authorship)                 reliability default 0.85
authorship= human 1.0 | tool:* 0.98 | model:* 0.95 | unknown 0.95        from generated_by
freshness = 0.8 + 0.2 x 0.5 ^ (age_days / half_life)                     half_life: slow 365, fast 45
            1.0 for volatility: static, an unknown verified date, or a historical query
status    = active 1.0 | draft 0.9 | superseded 0.5 | archived 0.3       1.0 for a historical query
type      = session 0.60 | research 0.85 | anything else 1.0
```

Frontmatter the layer reads (every key is optional):

```yaml
---
id: ref-pricing-2026
type: reference          # session and research notes are demoted
status: active           # active | draft | superseded | archived
tier: 1                  # 0 = primary source .. 4 = synthesis of synthesis
reliability: 0.95        # (0, 1]
generated_by: human      # human | tool:<name> | model:<name>
verified: 2026-10-01     # last date the facts were checked
volatility: fast         # static | slow | fast
sensitive: false         # true => never indexed
---
```

Every factor has its own switch in `[provenance]`, so each one can be measured alone.
`hrag query ... --explain` prints all factors and both ranks for every hit.

### Choosing rrf_k

RRF throws away score magnitude: with `k = 60` (the textbook value) the gap between rank 1 and
rank 5 is about 6 %, smaller than the provenance factors, so a weakly relevant but "trusted"
chunk could jump over the right answer. `k = 20` widens that gap to about 16 %. On the eval
below, `k = 60` with full provenance dropped MRR from 0.800 to 0.706 on the offline embedder.
The factor ranges were narrowed for the same reason: provenance is a tie-breaker between
relevant notes, not a second relevance signal.

## Evaluation

`hrag eval` indexes the sample vault and scores 15 labelled queries
([eval/queries.jsonl](eval/queries.jsonl)): 6 provenance conflicts (stale vs current runbook,
model-written chat summary vs reference page, two "active" price lists, superseded decision
record, research note vs decision), 4 keyword, 4 paraphrase and 1 query whose only answer is
an archived postmortem. A hit counts at note level; the clock is fixed at 2026-11-01.

**Offline hashing embedder** (`hrag eval --embedder hashing`, deterministic, the CI setting):

| Configuration | recall@1 | recall@3 | recall@5 | MRR |
|---|---:|---:|---:|---:|
| FTS5 only (BM25) | 0.733 | 1.000 | 1.000 | 0.867 |
| Dense only | 0.733 | 0.867 | 0.867 | 0.797 |
| Hybrid RRF, no provenance | 0.667 | 1.000 | 1.000 | 0.811 |
| Hybrid RRF + status filter | 0.600 | 0.933 | 0.933 | 0.767 |
| Hybrid RRF + full provenance | 0.667 | 0.933 | 0.933 | 0.800 |

**sentence-transformers/all-MiniLM-L6-v2** (`--embedder sentence-transformers --embed-model sentence-transformers/all-MiniLM-L6-v2`):

| Configuration | recall@1 | recall@3 | recall@5 | MRR |
|---|---:|---:|---:|---:|
| FTS5 only (BM25) | 0.733 | 1.000 | 1.000 | 0.867 |
| Dense only | 0.867 | 1.000 | 1.000 | 0.922 |
| Hybrid RRF, no provenance | 0.867 | 1.000 | 1.000 | 0.933 |
| Hybrid RRF + status filter | 0.867 | 0.933 | 0.933 | 0.900 |
| Hybrid RRF + full provenance | **0.933** | 0.933 | 0.933 | 0.933 |

**MiniLM + cross-encoder/ms-marco-MiniLM-L6-v2 reranker** (`--reranker cross-encoder`, `HRAG_RERANKER_MODEL=cross-encoder/ms-marco-MiniLM-L6-v2`):

| Configuration | recall@1 | recall@3 | recall@5 | MRR |
|---|---:|---:|---:|---:|
| FTS5 only (BM25) | 0.733 | 1.000 | 1.000 | 0.867 |
| Dense only | 0.667 | 1.000 | 1.000 | 0.822 |
| Hybrid RRF, no provenance | 0.733 | 1.000 | 1.000 | 0.867 |
| Hybrid RRF + status filter | 0.733 | 0.933 | 0.933 | 0.833 |
| Hybrid RRF + full provenance | **0.933** | 0.933 | 0.933 | **0.933** |

The default model, `BAAI/bge-m3`, was **not run** for these tables (2.2 GB of weights); the
code path is the same as for MiniLM.

How to read this:

- **On the 6 conflict queries provenance does what it is for.** MRR on that subset goes from
  0.833 to 1.000 with MiniLM and from 0.722 to 0.917 with the hashing embedder. The
  cross-encoder scores text, not sources: without provenance it puts the 2025 runbook, the
  chat summary, the research note and a retro session note first on four queries, and full
  provenance fixes all four.
- **The overall MRR barely moves**, because the one archived-gold query (q15) is lost: the
  status filter hides archived notes unless the question sounds historical, and "What went
  wrong in the login outage?" does not. That is the trade-off of a hard filter. Asking
  "what went wrong in the login outage at the time?" or passing `--historical` finds it.
- **On the offline hashing embedder, FTS5 alone beats every hybrid row.** The hashing
  "embedder" is lexical, so it adds noise rather than meaning, and fusing that noise moves the
  right note off rank 1 more often than it helps. It exists for tests and the demo, not for
  quality.
- These are 15 queries on 21 synthetic notes, and the multipliers were chosen while looking
  at them. Treat the tables as a reproducible illustration of the mechanism, not a benchmark.

## Configuration

`hrag.example.toml` lists every key with its default. Pick a file with `--config` or
`HRAG_CONFIG`, otherwise `./hrag.toml` is used if present and owned by you. Environment variables override the
file, and command-line flags override both:

| Variable | Meaning |
|---|---|
| `HRAG_VAULT` | folder of markdown notes (default: the bundled sample vault) |
| `HRAG_DATA_DIR` | index folder (default `.hrag`) |
| `HRAG_EMBEDDER` | `sentence-transformers` (default) or `hashing` |
| `HRAG_EMBED_MODEL` | any sentence-transformers model (default `BAAI/bge-m3`) |
| `HRAG_EMBED_DEVICE` | `auto`, `cpu`, `mps` or `cuda` |
| `HRAG_RERANKER` | `none` (default), `overlap`, `cross-encoder`, `flashrank` |
| `HRAG_RERANKER_MODEL` | cross-encoder model (default `BAAI/bge-reranker-v2-m3`) |
| `HRAG_PROVENANCE` | `0` turns the whole layer off |
| `HRAG_REFERENCE_DATE` | `YYYY-MM-DD` used as "today" by the freshness factor |
| `HRAG_SOCKET` | daemon socket path |

If `sentence-transformers` is not installed or the model cannot load, the embedder falls back
to `hashing` and logs a warning. An index built with one embedder is never queried with
another: the retriever notices the mismatch and degrades to keyword-only until you re-index.

## Daemon protocol

One JSON object per line over the socket:

```
-> {"op": "query", "query": "how long are backups kept?", "k": 5}
<- {"ok": true, "result": {"mode": "hybrid", "hits": [...], "timings": {...}, ...}}
-> {"op": "ping"}
<- {"ok": true, "op": "ping", "dense_ready": true, "heavy_busy": false, "served": 12}
```

The default socket is `$TMPDIR/hrag-<uid>/hrag.sock`. The daemon creates that folder with
mode 0700 and refuses a folder or socket owned by another user, and the client checks the
socket's owner and the folder before connecting. `--tcp host:port` accepts loopback addresses
only; the daemon then writes a random token (mode 0600) into the same private folder and
rejects every request that does not carry it, so other local users cannot query it either.
The reply carries an HMAC of a client nonce under that token, so the client also knows it
reached the real daemon. Requests are size-capped and time out after 30 s.
The index folder is created with mode 0700, and a folder owned by another user is refused.

One request at a time uses the model. Others wait up to `admit_wait_s` (5 s); a request that
waits longer, or arrives when `admit_max_waiters` (4) are already waiting, is answered
keyword-only with the reason in the result instead of piling up.

## Honest limits

- Brute-force cosine over a numpy matrix: fine for a personal or team notes archive (tens of
  thousands of chunks), not for millions; there is no ANN index.
- The frontmatter parser handles flat `key: value` pairs and inline lists only, not full YAML.
- The daemon and the index lock use Unix sockets and `fcntl`: macOS and Linux, not Windows.
- The historical-query detector is a list of English regexes.
- Provenance is only as good as the frontmatter. Notes without it get neutral factors.
- The eval set is small and synthetic, and the defaults were tuned on it (see above).
- This is a clean rewrite of a retriever I run over a private notes archive, not that code.
  Lessons from it shaped the defaults: in production, strong trust and freshness multipliers
  on top of RRF made answers worse, which is why the factors here are narrow and why
  `rrf_k` is 20.

## What this demonstrates

Building retrieval that is honest about its sources: hybrid dense plus keyword search fused
with RRF, a provenance layer whose every factor has a formula, a switch and a measured effect,
and a system that keeps answering when the model, the index or the machine is not cooperating.
It also shows the evaluation habit behind it: a labelled query set, an ablation table per
component, results reported even where they are unflattering (FTS5 beating the toy hybrid, the
archived-note miss), and failure modes covered by tests rather than described in prose.

## Dependencies and model licences

Checked on 2026-11-12 against PyPI package metadata and Hugging Face model cards. No model
weights are included in this repository.

| Component | Used for | Required? | Licence | Source |
|---|---|---|---|---|
| Python stdlib (`sqlite3` with FTS5, `tomllib`, `socketserver`) | storage, keyword search, config, daemon | yes | PSF-2.0 (SQLite itself is public domain) | python.org |
| numpy | dense matrix, cosine | yes | BSD-3-Clause (plus 0BSD, MIT, Zlib, CC0-1.0 for bundled parts) | PyPI 2.5.4 |
| pytest | tests | dev only | MIT | PyPI 9.2.0 |
| sentence-transformers | neural embedder, cross-encoder | `[models]` | Apache-2.0 | PyPI 6.3.0 |
| torch | runtime of the above | via `[models]` | Apache-2.0 AND BSD-3-Clause AND MIT and others (full SPDX expression in package metadata) | PyPI 2.15.0 |
| transformers, huggingface-hub | runtime of the above | via `[models]` | Apache-2.0 | PyPI 5.22.2, 2.4.0 |
| scikit-learn, scipy | runtime of the above | via `[models]` | BSD-3-Clause | PyPI 1.9.1, 1.18.1 |
| flashrank | FlashRank reranker | `[flashrank]` | Apache-2.0 | PyPI 0.2.10 |
| psutil | memory-pressure check | `[memory]` | BSD-3-Clause | PyPI 7.2.2 |
| **BAAI/bge-m3** | default embedding model | downloaded by user | MIT | model card |
| BAAI/bge-reranker-v2-m3 | default cross-encoder model | downloaded by user | Apache-2.0 | model card |
| sentence-transformers/all-MiniLM-L6-v2 | small embedder used in the eval | optional | Apache-2.0 | model card |
| cross-encoder/ms-marco-MiniLM-L6-v2 | small reranker used in the eval | optional | Apache-2.0 | model card |
| FlashRank `ms-marco-MultiBERT-L-12` | default FlashRank model | downloaded by flashrank | CC-BY-SA-4.0 (card of `prithivida/flashrank`, which hosts it) | model card |

Two notes. The FlashRank model is share-alike, unlike everything else here; if that matters
for your use, pick another FlashRank model or the cross-encoder. And the MS MARCO data the
`ms-marco` rerankers were trained on has its own non-commercial terms, which the model cards
above do not restate; check them before commercial use.

## Licence

MIT, see [LICENSE](LICENSE).
