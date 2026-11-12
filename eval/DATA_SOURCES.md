# Data sources of the evaluation set

| File | Source | Licence |
|---|---|---|
| `eval/questions.jsonl` | Written for this repository. Every question, gold answer and gold note id refers to the synthetic sample vault in `src/hybrid_rag/sample_vault/`. | MIT, as the rest of the repository |
| `src/hybrid_rag/sample_vault/*.md` | Written for this repository. "Quillmoor Labs" and "Tidewell" are fictional; the notes describe no real company, product, person or system. | MIT, as the rest of the repository |

No external documents are used: no public datasets, no web pages, no real notes, mail, chat logs,
customer or employer material, and no personal data of anyone.

## Format

One JSON object per line:

| Field | Meaning |
|---|---|
| `id` | stable id; the prefix mirrors the tag (`kw`, `pa`, `mh`, `sf`, `un`) |
| `question` | the query as a user would type it |
| `gold_notes` | note ids (frontmatter `id`) that answer the question; empty for unanswerable items |
| `gold_answer` | a short reference answer; empty for unanswerable items |
| `required_citations` | note ids a correct answer must cite (a subset of `gold_notes`) |
| `tag` | `keyword-heavy`, `paraphrase`, `multi-hop`, `stale-vs-fresh` or `unanswerable` |

Tag meanings:

- **keyword-heavy**: the question shares distinctive words with the answer passage.
- **paraphrase**: the question avoids the passage's words ("get their money back" for a refund policy).
- **multi-hop**: the answer needs two notes; both are gold and both must be cited.
- **stale-vs-fresh**: an older, superseded, lower-trust or model-written note competes with the current
  one. The gold note is the current one, except for the two questions phrased as historical
  ("previously", "at the time"), where the old note is the right answer.
- **unanswerable**: the vault does not contain the answer; the right behaviour is "no answer".

## Freezing

`eval/MANIFEST` holds the sha256 of `questions.jsonl`. `eval/run.py` refuses to run the default set
when the hash differs, and a test checks it. Changing the set means a new hash in a reviewed commit,
and old runs keep the hash they were scored with in their `metrics.json`.
