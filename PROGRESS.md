# Progress — Transaction NER assignment

Read this file (and `CLAUDE.md`) at the start of any new session before doing
anything else, then confirm with the user before starting the next phase.

## Status: Phase 4 complete; Phase 5 blocked on user go-ahead + submission endpoint details

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

**Phase 4 — local eval harness** (commit `9b6f196`)
- `fields.py::extract_fields`: tokens+tags -> the 4 scored field strings
  (non-contiguous spans concatenated in order -- documented as
  bag-of-words-safe since scoring is multiset-based).
- `eval_harness.py`: bag-of-words P/R/F1 (case/whitespace-insensitive) for
  counterparty/transaction_method/processor, BOTH micro (pooled) and macro
  (mean of per-example F1) variants since the server's aggregation choice
  is unspecified; presence-based P/R/F1 for recurring_flag. Final score =
  mean of the 4 field F1s (computed both ways).
- Verified against a hand-worked synthetic example (exact expected P/R/F1
  computed by hand in the test docstring): macro_f1=0.45, micro_f1=8/11,
  presence f1=0.5, all matched exactly. 9/9 new tests, 32/32 total.

**Phase 5 — calibration submission (IN PROGRESS, blocked)** (commit `17c52aa`)
- `infer.py`: `score_on_val()` (checkpoint -> eval_harness report on
  reconciled val) and `build_predictions_json()` (checkpoint -> validated
  predictions.json for test).
- Trained a real baseline (not the Phase 3 smoke run): full reconciled
  train/val, 3 epochs, CPU (~10 min), `logs/phase5_baseline_train.log`.
  val_loss 0.92 -> 0.14.
- Local val score (`logs/phase5_val_score.json`, gitignored):
  final_score_macro=0.969, final_score_micro=0.953. **Caveat**:
  recurring_flag reads as a trivial perfect 1.0 only because val has zero
  positive examples (predicts nothing, matches nothing present) -- this
  number carries NO information about real recurring_flag recall; only a
  live submission can measure that.
- `predictions.json` built for all 10,000 test ids and passed
  `validate_predictions` (exact id match, all 4 fields present, no
  nulls). Present locally but **not committed** -- treated as a working
  file until Phase 7/8 picks a final model.
- **Sanity-check flag**: test-set predicted field presence rates
  diverge from train's actual rates -- processor predicted-present on
  42% of test transactions vs. train's true 13.1%; counterparty
  predicted-present on 40% vs. train's true 66%. Vocabulary inspection
  (top predicted tokens) shows no degenerate failure (diverse, plausible
  tokens per field -- processor top tokens are real processor names like
  "intuit"/"paypal"/"venmo"), but counterparty's top predicted tokens
  include "withdrawal"/"external"/"deposit", which look more like
  BANK_SERVICE_EVENT vocabulary than counterparty names -- i.e. the
  3-epoch baseline plausibly confuses these two classes some of the
  time. Not treated as a blocking bug (pipeline is unit-tested
  end-to-end and the round-trip/shape checks all pass) -- flagged as a
  real baseline-quality issue for Phase 6 to address, and as a reason
  the calibration submission's job is exactly to tell us how much this
  matters for the real score.
- **BLOCKED**: have not called the real submission API. Two blockers:
  (1) hard rule requires explicit user go-ahead with the exact payload
  stated first (not yet asked/granted in-session at the time of writing);
  (2) the actual submission endpoint (URL/method/auth/payload shape) was
  never given anywhere in the original brief -- only the 3 dataset URLs
  were. Need to ask the user for this before Phase 5 can actually
  complete.

## Environment notes for whoever picks this up
- Python 3.12 venv at `.venv/` (created via `uv venv`), deps in
  `requirements.txt` (grouped by which phase first needs them — torch/
  transformers not yet installed, deferred to Phase 3).
- `uv pip install` needs `--system-certs` flag (TLS interception on this
  network breaks the bundled CA bundle).
- `git push` needs `GIT_SSL_NO_REVOKE=true` prefix (schannel revocation-check
  timeout, likely same interception). See CLAUDE.md "Environment quirks".
- Run tests: `.venv/Scripts/python.exe -m pytest -v` (12 passing as of Phase 1).

## Next: unblock Phase 5, then Phase 6
Need from the user before anything else:
1. Explicit go-ahead to spend a real submission attempt on the baseline
   `predictions.json` described above (exact local scores are in this file).
2. The actual submission API details (URL/method/auth/payload shape) --
   not present anywhere in the original brief, only the 3 dataset URLs were
   given.
Once both are in hand: call the endpoint once, compare the server's
per-field F1 to `logs/phase5_val_score.json`'s numbers, and adjust
eval_harness.py's micro/macro choice (and anything else that doesn't line
up) to match. Brief caps calibration at 1-2 attempts total.

After that: Phase 6 (model iteration) -- try a rule-based recurring_flag
detector (mandatory per EDA finding), address the counterparty/
BANK_SERVICE_EVENT confusion flagged above (more epochs / class weighting /
an upgrade path), compare each change against the baseline on the
now-calibrated local harness before keeping it.
