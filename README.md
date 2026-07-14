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
- [ ] Phase 1 — EDA (label distribution, scored-vs-unscored gap, inter-annotator agreement)
- [ ] Phase 2 — label reconciliation
- [ ] Phase 3 — baseline model (`jhu-clsp/ettin-encoder-32m` token classifier)
- [ ] Phase 4 — local eval harness
- [ ] Phase 5 — calibration submissions
- [ ] Phase 6 — model iteration
- [ ] Phase 7 — final inference & submission formatting
- [ ] Phase 8 — final submission(s) and writeup

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
