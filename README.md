# Transaction NER

Token classification pipeline for the "Transaction NER" assignment: given a
pre-tokenized bank transaction description, tag each token and extract four
scored fields — `counterparty`, `transaction_method`, `processor`,
`recurring_flag` — plus two unscored-but-annotated categories
(`bank_service_event`, `filler_word`) and separator/`O` tokens.

Scoring is macro-F1 averaged across the four scored fields on a held-out
test set, evaluated via a leaderboard with a limited number of submission
attempts. This repo is built to make each attempt count: a local evaluation
harness (`eval_harness.py`) is calibrated against the server's reported
score before it's trusted for model selection.

## Status

Work is being built in phases, each committed and tested independently.
See `SUBMISSIONS.md` for the calibration/improvement history once
submissions start.

- [x] **Phase 0** — repo setup, `data.py` ingestion (downloads + validates
      train/val/test against the documented schema)
- [x] **Phase 1** — EDA. Key findings (full writeup in `EDA_FINDINGS.md`):
      `I-RECURRING_FLAG` has **zero** occurrences in train/val (rule-based
      detection is mandatory, not optional); 3 of the 5 `annotator_id`s show
      **exact, zero-mismatch agreement** with each other and cover 100% of
      train/val ids, making reconciliation close to trivial (Phase 2); only
      13% of transactions contain a `PROCESSOR` span (thin-positive class).
- [x] **Phase 2** — label reconciliation: `reconcile_big_three()` (primary)
      takes the BIG_THREE_ANNOTATORS row as gold per id; `reconcile_duplication()`
      and `reconcile_token_tiebreak()` kept as documented alternatives.
- [x] **Phase 3** — baseline model: `model.py` (AutoModel + linear head,
      encoder swappable via `ModelConfig`), `tokenization.py` (subword/label
      alignment), `train.py` (plain PyTorch training loop), `predict.py`
      (batched inference, predictions realigned to original tokens). Smoke
      test (300 train / 100 val, 2 epochs, CPU) confirms the loop runs
      cleanly: train loss 1.99→0.77, val loss 0.92→0.61, and inference
      round-trips predicted tags back to the original token count exactly.
- [x] **Phase 4** — local eval harness: `eval_harness.py`, bag-of-words
      micro+macro P/R/F1 for counterparty/transaction_method/processor,
      presence P/R/F1 for recurring_flag.
- [x] **Phase 5** — calibration submission: real submission revealed
      macro_f1=0.4994, far below local val, despite test's true label
      distribution matching train closely (per response `support`
      counts). Traced to a case-sensitivity bug, not a metric mismatch.
      `eval_harness.py` calibrated to micro (pooled) aggregation, matching
      the response's `"metric": "token"` fields exactly.
- [x] **Phase 6** — model iteration: rule-based `recurring_flag` detector
      (`recurring.py`, mandatory per Phase 1 -- zero training signal
      exists). Diagnosed and disproved a suspected counterparty/
      BANK_SERVICE_EVENT confusion (`diagnostics.py`).
- [x] **Phase 7** — final inference: found and fixed the case-sensitivity
      bug (`tokenization.py::normalize_case` -- test is 51% UPPERCASE vs
      train/val's ~28%, and the tokenizer fragments all-caps merchant
      names into junk subwords). Retrained, rebuilt `predictions.json`.
- [x] **Phase 8** — final submission: macro_f1 = **0.8975** (up from
      0.4994 pre-fix), attempt 2/20. Per-field F1: counterparty 0.90,
      transaction_method 0.94, processor 0.82, recurring_flag 0.92
      (recall 1.0).

## Repo layout

```
config.py          Seeds, tag vocabulary, data URLs, expected dataset shapes
data.py            Download + parse train/val/test.jsonl, joined/grouped by id
tests/             pytest unit/smoke tests, one file per module
data_cache/        Cached raw .jsonl downloads (gitignored, re-fetchable)
requirements.txt   Deps, grouped by which phase first needs them
```

More modules (`reconcile.py`, `model.py`, `train.py`, `eval_harness.py`,
`infer.py`) are added as their phases land.

## Setup

```bash
uv venv --python 3.12
uv pip install --system-certs -r requirements.txt   # --system-certs needed behind this network's TLS proxy
.venv/Scripts/python.exe data.py                     # downloads + validates all 3 splits
.venv/Scripts/python.exe -m pytest -v
```

## Key data facts (see `config.py::EXPECTED_SHAPES` for the enforced version)

- `train.jsonl`: 20,000 rows = 10,000 unique transaction ids × 2 annotators.
- `val.jsonl`: 2,000 rows = 1,000 unique ids × 2 annotators.
- `test.jsonl`: 10,000 rows, one per unique id, no labels.
- For every train/val id, both annotator rows share an **identical** `tokens`
  array (fixed tokenization) but can have **different** `ner_tags`.
- The dataset API reshuffles line order on every download — everything in
  this repo joins/groups on `id`, never on row position (`data.py` asserts
  this holds on every load).
