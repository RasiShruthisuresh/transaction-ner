# Progress — Transaction NER assignment

Read this file (and `CLAUDE.md`) at the start of any new session before doing
anything else, then confirm with the user before starting the next phase.

## Status: Phase 6 (partial) done; Phase 5 submission still blocked pending endpoint details

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

**Phase 6 (partial) — model iteration** (commit pending)
- `recurring.py`: rule-based `recurring_flag` detector (`detect_recurring`),
  regex over the same keyword families EDA used to measure the ~2.3% base
  rate (`recur|preauth|subscri|auto(debit|pay|approv|renew)`,
  case-insensitive). Wired into `infer.py::predict_fields_for_df` only --
  overrides the model's (always-empty) recurring_flag prediction, but
  does NOT touch how gold fields are extracted for scoring (gold must
  keep reflecting the real annotation gap, not our rule's guess, or
  local eval would trivially agree with itself). 12/12 new tests.
- `diagnostics.py`: token-level confusion matrix + per-tag P/R/F1 over
  all 8 tags (`token_confusion`, `per_tag_precision_recall`,
  `diagnose_checkpoint`), independent of eval_harness.py's field-level
  scoring. Built to quantify the counterparty/BANK_SERVICE_EVENT
  confusion Phase 5 flagged qualitatively. 3/3 new tests.
- **Investigated the Phase 5 counterparty/BANK_SERVICE_EVENT confusion
  and it did not replicate on labeled data.** That flag came from
  eyeballing vocabulary in unlabeled *test* predictions only. Running
  `diagnostics.py` against the baseline checkpoint on **val** (which has
  real gold labels) found **zero** tokens where true
  `I-BANK_SERVICE_EVENT` got predicted as `I-COUNTERPARTY_NAME`
  (`logs/phase6_diagnose_baseline.log`), and per-field predicted-presence
  rates on val match gold almost exactly (counterparty 67.6% predicted
  vs 65.2% gold; processor 13.8% vs 13.5% gold; transaction_method 69.7%
  vs 70.9% gold). So on labeled data there was nothing to fix.
- **The real open question is test-set presence-rate divergence, and
  it's still unexplained.** Test's predicted-present rates (from the
  existing baseline, before this session's changes) are far from train's
  true rates: counterparty 39.8% predicted vs train's true 66.25%,
  processor 42.1% vs train's true 13.10%, transaction_method 57.9% vs
  train's true 71.69%. Checked and ruled out as explanations: bank
  coverage (only 7/10,000 test rows have a bank never seen in train),
  token OOV rate against train (test rows: 68.4% contain >=1 OOV token;
  val rows: 68.3% -- essentially identical, and val still scores well),
  sequence length (test mean 8.17 vs train 8.12 tokens), and
  transaction_type mix (test 75.3%/24.7% DEBIT/CREDIT vs train
  75.6%/24.4%). None of the checkable inputs explain the divergence --
  flagged as real but **only resolvable by a live submission**, not
  something to chase further with local-only tweaking.
- **Redirected effort accordingly**: instead of "fixing" a confusion that
  doesn't reproduce on labeled data, attempted the cheap, well-motivated
  experiment that was still open -- retraining longer (baseline was only
  3 epochs; val_loss was still dropping fast, 0.92->0.14). This did NOT
  complete: first attempt hung indefinitely mid-setup on Hugging Face Hub
  metadata calls (likely the same TLS-interception issue noted below,
  triggered on the larger model.safetensors GET rather than the small
  HEAD requests other runs tolerated); killed and retried with
  `HF_HUB_OFFLINE=1`/`TRANSFORMERS_OFFLINE=1` (model was already locally
  cached, so this is safe and much faster -- worth using by default for
  reruns going forward); second attempt was still mid-first-epoch when
  the user's time budget (5 min) ran out, so it was killed with **no
  checkpoint produced**. **Decision: kept `checkpoints/baseline.pt` (the
  3-epoch Phase 5 model) as the Phase 6 model** -- nothing was lost, and
  diagnostics gave no evidence the baseline needed fixing in the first
  place. Retraining longer remains a legitimate thing to try later with
  more time, using `HF_HUB_OFFLINE=1` from the start.
- Rebuilt `predictions.json` (baseline checkpoint + recurring_flag rule).
  `recurring_flag` now fires on 273/10,000 test transactions (2.7%,
  matching the ~2.3-2.6% keyword base rate from EDA across all splits --
  confirms the rule is working as intended). Other 3 fields unchanged
  (same model, same predictions as Phase 5).
- Re-scored on val with the rule wired in
  (`logs/phase6_val_score.json`): `final_score_micro` dropped from the
  Phase 5 number (0.953) to **0.703**, and `final_score_macro` from 0.969
  to **0.719**. **This is expected, not a regression** -- the rule fires
  on 26/1000 val examples (matching val's known ~2.6% keyword rate
  exactly), and since val's true recurring_flag labels are always empty
  (the Phase 1 label-gap finding), every one of those 26 hits scores as
  a false positive locally, cratering recurring_flag's local F1 from a
  meaningless 1.0 (empty-matches-empty on every example) down to a real
  0.0. The other 3 fields' scores are unchanged. **Only a live submission
  can say whether the rule's real recall on genuinely recurring test
  transactions outweighs this local-only false-positive cost.**

**Phase 5 real calibration submission** (2026-07-14, attempt 1/20, via
Colab per assignment's `submit_predictions()` mechanism,
`POST http://3.108.8.61:8990/submit`, multipart form-data:
predictions.json + name/email/roll_number/college): real macro_f1 =
**0.4994**. Full response saved to
`logs/phase5_calibration_submission_response.json` (gitignored).
Per-field: counterparty f1=0.292 (precision 0.42 recall 0.22),
transaction_method f1=0.52, processor f1=0.26 (precision 0.16 -- badly
over-firing), recurring_flag f1=0.92 (**recall 1.0** -- the rule-based
detector worked great). `support` counts confirm test's TRUE label
distribution matches train's closely (counterparty support/10000=65.8%
vs train 66.25%; processor 13.6% vs 13.10%; method 71.6% vs 71.69%) --
so the gap vs local val (~0.94-0.97) is a real MODEL problem, not a
label-distribution or metric-definition mismatch.

**Root cause found: case-sensitivity bug.** Test tokens are 51.1%
UPPERCASE vs train/val's ~28% (val matches train almost exactly, which
is why val never exposed this). Tokenizer is case-sensitive and
fragments all-caps merchant names into junk subwords
(`tokenize("PAYPAL")` -> `['PA','YP','AL']` vs `tokenize("Paypal")` ->
`['Pay','pal']`). This plausibly explains most of the counterparty/
processor collapse on test.

**Phase 6/7 fix**: `tokenization.py::normalize_case()` lowercases tokens
right before they hit the tokenizer (both `TransactionDataset` in
training and `predict.py::predict_tags` at inference) -- original-case
tokens still used everywhere else (label alignment, `extract_fields`
output). Retrained (`checkpoints/phase7_lowercased.pt`, only **2 of 3
planned epochs** completed under a hard time budget -- val_loss
0.29->0.21, still improving when stopped). Rebuilt `predictions.json`:
predicted test presence rates jumped from 39.8%/57.9%/42.1%
(counterparty/method/processor) to **64.8%/70.5%/12.3%** -- now closely
tracking the true rates above (65.8%/71.6%/13.6%). Strong local evidence
the fix works; a second real submission is how to confirm it on the
actual score.

**Phase 5 eval_harness.py calibration** (done): the response's
`"metric": "token"` fields back-solve exactly to pooled precision/recall
over token counts (not per-example-averaged) -- confirms
`final_score_micro` (not macro) is the right local proxy;
`report["final_score"]` now aliases it. `"metric": "presence"` for
recurring_flag confirms `presence_score` was already correct. Still
open: empty-vs-empty convention (low priority, didn't explain the gap).

**Phase 8 — final submission (attempt 2/20, 2026-07-14): macro_f1 = 0.8975**
(up from 0.4994 pre-fix). Per-field F1: counterparty 0.90, transaction_method
0.94, processor 0.82, recurring_flag 0.92 (recall 1.0). Confirms the
case-normalization fix (Phase 7) was the right call. `best_macro_f1`
= 0.8975. 18 attempts remaining. Response saved to
`logs/phase8_final_submission_response.json` (gitignored).

## Environment notes for whoever picks this up
- Python 3.12 venv at `.venv/` (created via `uv venv`), deps in
  `requirements.txt` (grouped by which phase first needs them — torch/
  transformers not yet installed, deferred to Phase 3).
- `uv pip install` needs `--system-certs` flag (TLS interception on this
  network breaks the bundled CA bundle).
- `git push` needs `GIT_SSL_NO_REVOKE=true` prefix (schannel revocation-check
  timeout, likely same interception). See CLAUDE.md "Environment quirks".
- Run tests: `.venv/Scripts/python.exe -m pytest -v` (12 passing as of Phase 1).

## Next: unblock Phase 5, then finish Phase 6
Still need from the user before Phase 5 can complete:
1. Explicit go-ahead to spend a real submission attempt on
   `predictions.json` (baseline model + rule-based recurring_flag;
   current local scores are in `logs/phase6_val_score.json`, see above
   for why the recurring_flag component looks artificially low locally).
2. The actual submission API details (URL/method/auth/payload shape) --
   not present anywhere in the original brief, only the 3 dataset URLs were
   given. Still not provided as of this session.
Once both are in hand: call the endpoint once, compare the server's
per-field F1 to `logs/phase6_val_score.json`'s numbers (especially
recurring_flag's real recall, which cannot be measured locally at all),
and adjust eval_harness.py's micro/macro choice to match. Brief caps
calibration at 1-2 attempts total.

Remaining Phase 6 ideas, not yet done (none blocked -- just not reached):
- Retry the longer-training experiment (10 epochs) with
  `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1` from the start and enough
  wall-clock time (~35 min estimated for 10 epochs on CPU); compare
  against baseline.pt on the local harness before keeping it.
- EDA finding #5's auxiliary-feature idea: prepend `bank` +
  `transaction_type` as lightweight context, compare against baseline.
- The test-set presence-rate divergence noted above is still
  unexplained and worth another look if a real submission score comes
  back much lower than the local harness predicts.
