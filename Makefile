# One-command demo: builds ./.venv with numpy on first use, then runs the nine scripted scenarios
# on the synthetic sample vault. No network after the numpy wheel, no keys, no model download.
demo: .venv/.ready
	PYTHONPATH=src .venv/bin/python -m hybrid_rag demo

.venv/.ready: requirements.txt
	python3 -m venv .venv
	.venv/bin/pip install --quiet -r requirements.txt
	touch .venv/.ready

# Offline evaluation. PYTHON can point at any interpreter with requirements-dev.txt installed.
PYTHON ?= python3
SEED ?= 42

.PHONY: demo eval eval-minilm eval-bge-m3 results test

# Default profile: hashing embedder + lexical overlap reranker. No network, no keys, no downloads.
eval:
	PYTHONPATH=src $(PYTHON) -m eval.run --seed $(SEED)

# Optional profiles: need `pip install -e ".[models]"`; download weights from Hugging Face once (no key).
eval-minilm:
	PYTHONPATH=src $(PYTHON) -m eval.run --seed $(SEED) --allow-network \
		--embedder sentence-transformers --embed-model sentence-transformers/all-MiniLM-L6-v2 \
		--reranker cross-encoder --reranker-model cross-encoder/ms-marco-MiniLM-L6-v2

eval-bge-m3:
	PYTHONPATH=src $(PYTHON) -m eval.run --seed $(SEED) --allow-network \
		--embedder sentence-transformers --embed-model BAAI/bge-m3 \
		--reranker cross-encoder --reranker-model cross-encoder/ms-marco-MiniLM-L6-v2

# Rebuild RESULTS.md and the README table from the committed runs without a new run.
results:
	PYTHONPATH=src $(PYTHON) -m eval.report

test:
	$(PYTHON) -m pytest -q
