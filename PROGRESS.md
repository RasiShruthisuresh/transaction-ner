# Progress — Transaction NER assignment

Read this file (and `CLAUDE.md`) at the start of any new session before doing
anything else, then confirm with the user before starting the next phase.

## Status: Phase 1 complete, ready for Phase 2 (label reconciliation)

## What's done

**Phase 0 — repo setup + ingestion** (commit `7740c8f`)
- Git repo created, pushed to `github.com/RasiShruthisuresh/transaction-ner` (main).
- `config.py`: seeds, tag vocabulary, dataset URLs, expected corpus shapes.
- `data.py`: downloads+caches train/val/test.jsonl, parses to DataFrames,
  validates schema. Verified against the live API: train 20000/10000,
  val 2000/1000, test 10000/10000, all ids/tokens/tags matching spec.

**Phase 1 — EDA** (commit `52153f3`)
- `eda.py` + `EDA_FINDINGS.md`. Two findings that drive everything downstream:
  1. **`I-RECURRING_FLAG` has zero occurrences in train or val**, even though
     the underlying keywords ("recurring", "preauthorized", etc.) appear in
     ~2.3% of transactions across all three splits — they're just labeled `O`
     or folded into other tags. No supervised signal exists for this class at
     all → **rule-based detection is mandatory** for this field, decided now
     rather than discovered after a wasted training run.
  2. **3 of 5 `annotator_id`s show exact, zero-mismatch agreement with each
     other** (verified exhaustively across ~48k train tokens + val), and
     together cover 100% of train/val ids (the other 2 annotators never
     co-occur with each other). These three ("big three": `ann_78009b`,
     `ann_928994`, `ann_f2aee8`) are being treated as one deterministic
     labeling process, not independent human raters. → **Phase 2
     reconciliation strategy: prefer the big-three row over the minority
     pair's row**, rather than duplication-as-augmentation or per-token
     tie-breaking (both alternatives from the brief were considered and
     rejected — see EDA_FINDINGS.md §4 for the full reasoning).
- Also found: `processor` is a thin-positive class (only 13.1% of
  transactions have one — risk of under-prediction, plan to
  oversample/upweight); `transaction_type` and `bank` both show usable
  correlation with field presence/vocabulary (bank mostly for high-volume
  banks given a long tail of 547 banks).

## Environment notes for whoever picks this up
- Python 3.12 venv at `.venv/` (created via `uv venv`), deps in
  `requirements.txt` (grouped by which phase first needs them — torch/
  transformers not yet installed, deferred to Phase 3).
- `uv pip install` needs `--system-certs` flag (TLS interception on this
  network breaks the bundled CA bundle).
- `git push` needs `GIT_SSL_NO_REVOKE=true` prefix (schannel revocation-check
  timeout, likely same interception). See CLAUDE.md "Environment quirks".
- Run tests: `.venv/Scripts/python.exe -m pytest -v` (12 passing as of Phase 1).

## Next: Phase 2 — label reconciliation
Plan (per EDA_FINDINGS.md §"How this changes the training strategy"):
- Primary strategy: for each train/val id, take the annotation row from a
  "big three" annotator (`config.BIG_THREE_ANNOTATORS`) as gold. Since every
  id has ≥1 big-three annotator and all big-three pairs already agree
  exactly, this needs no per-token tie-breaking logic.
- Keep the duplication-as-augmentation and token-level-tie-break code paths
  available (not deleted) so the alternative can be inspected/compared, per
  the brief's request — but they are not the primary path.
- Unit test to write: reconciled dataset has exactly 10,000 rows (one per
  train id) / 1,000 rows (val), tokens match the source id's tokens exactly,
  and every reconciled row's annotator_id is in BIG_THREE_ANNOTATORS.
