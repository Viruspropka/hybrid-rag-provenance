# Screen recording plan: hybrid-rag-provenance

Target length 2:30 to 3:00, one take. Everything shown is the synthetic sample vault; do not
open any real notes, accounts or browser tabs. The terminal recording in `docs/demo.gif` is
already rendered; this plan is for a narrated version that only the repository owner can record.

## Setup (before pressing record)

- Terminal 110 columns by 36 rows, font 16 pt, dark theme, empty prompt, no history visible.
- `cd` into the repository, run `make demo` once so `.venv` exists, then `clear`.
- Close notifications, chat apps and other windows. Screen: terminal only.

## Shot list

| Time | Shot | Type | Say (one line) |
|---|---|---|---|
| 0:00 | Title: README top in the editor, scroll to the architecture diagram | scroll | "This is a retriever for markdown notes that ranks by relevance first and source quality second." |
| 0:20 | Terminal: `make demo` | type | "One command, offline, on a made-up company's notes." |
| 0:30 | Scenario 1 and 2 scroll by | pause 3 s | "Dense and keyword results are merged by reciprocal rank; each hit shows its citation." |
| 0:55 | Scenario 3, both blocks | pause 6 s | "Two price lists, both marked active. With provenance off they tie; with it on the newer verified one wins." |
| 1:30 | Scenario 5, historical query | pause 4 s | "Old notes stay hidden unless you ask about the past." |
| 1:50 | Scenario 7, LOW CONFIDENCE | pause 4 s | "When nothing fits, it says so instead of guessing." |
| 2:10 | Scenario 8, degraded mode | pause 5 s | "I delete the dense index; keyword search still answers and tells you why." |
| 2:35 | `demo OK`, then `make eval` summary, `RESULTS.md` table | pause 3 s | "Every number in the README comes from a committed run log." |
| 2:50 | End on the README "What this proves" list | hold 3 s | "Code and tests are in the repo." |

## Notes

- If a scenario scrolls too fast, press Ctrl-S to pause output and Ctrl-Q to resume, or
  record with `python -m hybrid_rag demo | less -R`.
- Owner-only steps: the voice-over, any face-on-camera clip, and uploading to a video site or
  social network. None of that is done here.
