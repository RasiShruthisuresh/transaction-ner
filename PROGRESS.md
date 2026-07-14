# Progress — Transaction NER assignment

Read this file (and `CLAUDE.md`) at the start of any new session before doing
anything else, then confirm with the user before starting the next phase.

## Status: Phase 3 complete, ready for Phase 4 (local eval harness)

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

**Phase 2 — label reconciliation** (commit `c2ef24e`)
- `reconcile.py`: `reconcile_big_three()` is primary (one row/id from a
  BIG_THREE_ANNOTATORS member; asserts the EDA coverage finding holds).
  `reconcile_duplication()` and `reconcile_token_tiebreak()` kept as
  documented, callable alternatives (not deleted) per the brief.
- Verified: 7/7 pytest, including a real-data check that `reconcile_big_three`
  produces exactly 10000/1000 rows for train/val with tokens matching source
  exactly and `annotator_id` always in BIG_THREE_ANNOTATORS.

**Phase 3 — baseline model** (commit pending, see below)
- `model.py`: `TokenClassifier` = `AutoModel` (encoder swappable via
  `ModelConfig.model_name`, default `jhu-clsp/ettin-encoder-32m`, a
  ModernBERT-arch 31.9M-param encoder) + `Dropout` + `Linear` head. Built as
  a manual head rather than `AutoModelForTokenClassification` because not
  every candidate encoder is guaranteed to have a registered token-classification
  head class.
- `tokenization.py`: `align_labels()` puts the label on each original
  token's FIRST subword only, -100 (ignored) elsewhere — standard HF
  recipe, chosen so loss isn't inflated by tokens that happen to split into
  more subpieces, and so inference only needs to read one prediction per
  original token.
- `train.py`: plain PyTorch loop (not HF Trainer, for transparency/fewer
  moving parts), CLI-configurable (encoder, lr, batch size, epochs,
  `--max-train-examples`/`--max-val-examples` for smoke tests), logs to
  `--log-file` under `logs/` per CLAUDE.md.
- `predict.py`: batched inference, realigns predictions back to one tag per
  original token via each token's first-subword position (symmetric with
  `align_labels`).
- Verified: smoke test (300 train / 100 val, 2 epochs, CPU,
  `logs/phase3_smoke.log`) — loop runs cleanly, train loss 1.99→0.77, val
  loss 0.92→0.61. Followed by an inference sanity check loading the smoke
  checkpoint and confirming predicted tag lists match original token counts
  exactly on 5 val examples, with non-degenerate (varied) predicted tags.
  23/23 pytest passing overall (added `tests/test_tokenization.py` for the
  alignment logic specifically).
- Not yet trained on full data — that's deferred to Phase 6 (model
  iteration) where baseline-vs-upgrade comparisons happen together, since
  Phase 4 (local eval harness) needs to exist first to actually score them.

## Environment notes for whoever picks this up
- Python 3.12 venv at `.venv/` (created via `uv venv`), deps in
  `requirements.txt` (grouped by which phase first needs them — torch/
  transformers not yet installed, deferred to Phase 3).
- `uv pip install` needs `--system-certs` flag (TLS interception on this
  network breaks the bundled CA bundle).
- `git push` needs `GIT_SSL_NO_REVOKE=true` prefix (schannel revocation-check
  timeout, likely same interception). See CLAUDE.md "Environment quirks".
- Run tests: `.venv/Scripts/python.exe -m pytest -v` (12 passing as of Phase 1).

## Next: Phase 4 — local eval harness
Must mirror the server metric as closely as possible before it's trusted for
model selection (see original brief step 5):
- Bag-of-words token-level precision/recall/F1 (case/whitespace-insensitive)
  for counterparty/transaction_method/processor, comparing predicted vs gold
  token multisets per transaction. Build BOTH micro (pooled counts) and
  macro (mean of per-example F1) variants since the server's aggregation
  choice isn't specified.
- Presence-based binary F1 for recurring_flag (non-empty prediction vs
  non-empty gold) — note per EDA_FINDINGS.md §2 that val has zero positive
  recurring_flag examples, so local recall for a rule-based detector can't
  be measured against val at all; only real signal comes from a Phase 5
  calibration submission.
- Final score = plain average of the four field F1s.
- Test with a hand-constructed synthetic example with known correct P/R/F1,
  and assert the code reproduces it exactly — this phase's "test" is that
  assertion passing.
- Decide + document how val (also 2-annotator) gets scored: against the
  reconciled (big-three) labels is the natural choice given Phase 2's
  finding, but note it in the harness code explicitly since it's a real
  design decision a reviewer will ask about.
