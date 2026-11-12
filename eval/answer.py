"""Deterministic extractive answerer and a rule-based answer checker.

The answerer stands in for an LLM so the whole evaluation runs offline and
gives the same result every time. It reads the top retrieved chunks, splits
them into sentences, and returns the one or two sentences that cover most of
the question's content terms, citing the note each sentence came from. It
abstains ("no answer") when the retriever flags low confidence or when no
sentence covers at least ``ABSTAIN_BELOW`` of the question terms.

The checker scores any answer of the form ``{"abstained", "sentences": [{"text",
"note_id"}], "citations"}``, so a real LLM answer can be scored the same way:

(a) every cited note id is in the retrieved set,
(b) citation precision (cited notes that are gold) and citation recall
    (required citations that were cited),
(c) a FAITHFULNESS PROXY: a sentence counts as supported when at least
    ``SUPPORT_THRESHOLD`` of its content tokens occur in one cited passage.
    This is token overlap, not entailment: it catches invented sentences and
    wrong citations, not a sentence that reuses the passage's words to say
    something false. For the extractive answerer it is a sanity check (it
    copies sentences, so it should be 1.0); it matters for generated answers.

All thresholds were fixed before the first run and were not tuned on the set;
the abstention threshold is reported as a sweep instead (``SWEEP``).
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from hybrid_rag.text import content_terms, light_stem, tokens

CONTEXT_HITS = 5          # chunks the answerer may read (the "retrieved set")
MAX_SENTENCES = 2
ABSTAIN_BELOW = 0.5       # best sentence must cover >= half of the question's content terms
SECOND_MIN_NEW_TERMS = 2  # a second sentence must add >= 2 question terms the first missed
SUPPORT_THRESHOLD = 0.6   # faithfulness proxy: share of sentence content tokens found in a cited passage

# (abstain_below, use the retriever's low-confidence flag) pairs reported in RESULTS.md
SWEEP = ((0.0, False), (0.25, False), (0.34, False), (0.5, False),
         (0.0, True), (0.25, True), (0.34, True), (0.5, True))

_SENT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9`\"(])")


def _stems(text: str) -> set[str]:
    return {light_stem(t) for t in content_terms(text)}


def split_sentences(chunk_text: str) -> list[str]:
    """Sentences of a chunk, markdown heading lines dropped."""
    body = " ".join(line.strip() for line in chunk_text.splitlines()
                    if line.strip() and not line.lstrip().startswith("#"))
    return [s.strip() for s in _SENT_RE.split(body) if s.strip()]


@dataclass
class Passage:
    note_id: str
    chunk_id: str
    text: str


def answer(question: str, passages: list[Passage], low_confidence: bool, *,
           abstain_below: float = ABSTAIN_BELOW, use_retriever_flag: bool = True) -> dict:
    """Extract an answer from ``passages`` (best first) or abstain.

    ``abstain_below`` and ``use_retriever_flag`` exist for the threshold sweep in
    RESULTS.md; the headline numbers use the defaults.
    """
    q = _stems(question)
    context = passages[:CONTEXT_HITS]
    base = {"retrieved": list(dict.fromkeys(p.note_id for p in context))}
    low_confidence = low_confidence and use_retriever_flag
    if not q or not context or low_confidence:
        return base | {"abstained": True, "reason": "retriever low confidence" if low_confidence else "no context",
                       "sentences": [], "citations": []}
    cands = []   # (coverage, -hit_rank, -sentence_index, stems, sentence, passage)
    for rank, p in enumerate(context):
        for i, s in enumerate(split_sentences(p.text)):
            st = _stems(s)
            cands.append((len(q & st) / len(q), -rank, -i, st, s, p))
    if not cands:
        return base | {"abstained": True, "reason": "no sentences", "sentences": [], "citations": []}
    cands.sort(key=lambda c: (c[0], c[1], c[2]), reverse=True)
    best = cands[0]
    if best[0] < abstain_below:
        return base | {"abstained": True, "reason": f"best sentence covers {best[0]:.2f} of the question",
                       "sentences": [], "citations": []}
    chosen = [best]
    covered = q & best[3]
    for c in cands[1:]:
        if len(chosen) >= MAX_SENTENCES:
            break
        if c[4] == best[4]:
            continue
        if len((q & c[3]) - covered) >= SECOND_MIN_NEW_TERMS:
            chosen.append(c)
            covered |= q & c[3]
    sentences = [{"text": c[4], "note_id": c[5].note_id, "chunk_id": c[5].chunk_id} for c in chosen]
    return base | {"abstained": False, "reason": "", "sentences": sentences,
                   "citations": list(dict.fromkeys(s["note_id"] for s in sentences)),
                   "best_coverage": round(best[0], 4)}


def sentence_support(sentence: str, passage_text: str) -> float:
    """Share of the sentence's content tokens that occur in the passage (0..1)."""
    s = {light_stem(t) for t in content_terms(sentence)}
    if not s:
        return 1.0
    p = {light_stem(t) for t in tokens(passage_text)}
    return len(s & p) / len(s)


def check(ans: dict, passages: list[Passage], gold_notes: list[str], required: list[str],
          gold_answer: str) -> dict:
    """Rule-based scores for one answer. ``passages`` = the retrieved set given to the answerer."""
    retrieved = {p.note_id for p in passages}
    cited = list(ans.get("citations", []))
    answerable = bool(gold_notes)
    out: dict = {"answerable": answerable, "abstained": bool(ans.get("abstained"))}
    if not answerable:
        out["correct_abstention"] = out["abstained"]
        return out
    if out["abstained"]:
        out |= {"citations_in_retrieved": None, "citation_precision": None, "citation_recall": 0.0,
                "faithfulness": None, "answer_term_recall": 0.0}
        return out
    gold, req = set(gold_notes), set(required)
    out["citations_in_retrieved"] = all(c in retrieved for c in cited)
    out["citation_precision"] = len([c for c in cited if c in gold]) / len(cited) if cited else 0.0
    out["citation_recall"] = len(req & set(cited)) / len(req) if req else 1.0
    supports = []
    for s in ans.get("sentences", []):
        texts = [p.text for p in passages if p.note_id in set(cited)]
        supports.append(max((sentence_support(s["text"], t) for t in texts), default=0.0))
    out["sentence_support"] = [round(x, 4) for x in supports]
    out["faithfulness"] = (sum(x >= SUPPORT_THRESHOLD for x in supports) / len(supports)) if supports else 0.0
    g = _stems(gold_answer)
    text = " ".join(s["text"] for s in ans.get("sentences", []))
    out["answer_term_recall"] = len(g & {light_stem(t) for t in tokens(text)}) / len(g) if g else 0.0
    return out
