# Phase 1 — EDA Findings

All numbers below are reproducible by running `python eda.py`, which writes the
underlying tables to `eda_outputs/*.csv` and two summary plots. Analysis is run
on **train** (10,000 transactions / 20,000 annotation rows); **val** is spot-checked
for consistency where noted, since it's held out for model selection, not for
characterizing the label distribution.

## 1. Token-level tag frequency: scored vs unscored

| tag | token count | token share | scored? |
|---|---:|---:|:---:|
| O | 76,533 | 47.1% | no |
| I-COUNTERPARTY_NAME | 27,690 | 17.1% | **yes** |
| I-TRANSACTION_METHOD | 19,649 | 12.1% | **yes** |
| I-SEPARATOR_PUNCTUATION | 16,984 | 10.5% | no |
| I-BANK_SERVICE_EVENT | 13,127 | 8.1% | no |
| I-FILLER_WORD | 5,111 | 3.1% | no |
| I-PROCESSOR | 3,268 | 2.0% | **yes** |
| I-RECURRING_FLAG | **0** | **0.0%** | **yes** |

Only **31.2%** of all tokens carry a scored tag; the rest is structural
(separators), bank-side noise (`BANK_SERVICE_EVENT`, `FILLER_WORD`), or `O`.
A model trained with plain per-token cross-entropy over all 8 classes will
spend most of its gradient on tags that don't affect the leaderboard score at
all — motivating either per-class loss weighting toward the 4 scored tags, or
at minimum not treating overall token accuracy as a proxy for the metric we
actually care about (this is why Phase 4 builds a dedicated harness instead of
trusting training-loop accuracy).

## 2. `I-RECURRING_FLAG` has **zero** occurrences in train or val — this is the central finding

Grepping `metadata.original_description` for recurring-related keywords
(`recur`, `preauth`, `subscri`, `auto(debit|pay|approv|renew)`) finds them in
**229/10,000 train ids (2.29%)**, **26/1,000 val ids (2.6%)**, and
**228/10,000 test ids (2.28%)** — a consistent ~2.3% base rate across splits.
So the concept is present in the raw text at a stable rate. But inspecting
what tag those exact keyword tokens received in train/val: **never**
`I-RECURRING_FLAG`. They're overwhelmingly `O` (e.g. `"Recurring"`,
`"Preauthorized"`, `"Recur"` all tagged `O`), occasionally folded into
`I-BANK_SERVICE_EVENT` (e.g. `"PREAUTHORIZED"`, `"AUTOPAYBUS"`) or
`I-TRANSACTION_METHOD` (e.g. `"PREAUTHPMT"`).

**Consequence:** a token classifier trained on train.jsonl has *zero* positive
gradient signal for this class — it is structurally incapable of learning to
predict `I-RECURRING_FLAG` from supervised training data alone, no matter how
it's weighted or oversampled. This isn't a "rare class, upweight it" problem;
there is nothing to weight. Per the brief's own guidance (recurring-flag
vocabulary is "close to a closed vocabulary"), **rule-based keyword detection
is not an optional upgrade for this field — it is the only viable approach**,
and the model's job for this field reduces to "don't actively contradict the
rule." This is decided now, in Phase 1, rather than discovered after wasting
a training run on it.

**Side effect for Phase 4 (local eval harness):** since val also has zero
positive `recurring_flag` labels, local recall for a rule-based detector can't
be measured against val at all — only precision-ish sanity checks (does the
rule fire on innocuous text?) and the keyword-hit-rate proxy above are
possible locally. Real recall feedback for this field can only come from a
calibration submission against the server (Phase 5). This is flagged
explicitly so a "perfect" local recurring_flag score is never mistaken for
evidence the field is solved.

## 3. Token share vs transaction-level presence (the imbalance the metric actually cares about)

`recurring_flag` is scored by **presence** (non-empty prediction vs non-empty
gold), so what matters for it is the fraction of *transactions* containing the
field, not the fraction of *tokens*. Token share and transaction-level
presence diverge substantially for the other three fields too:

| field | token share | row presence rate | txn presence rate (≥1 annotator) | presence ÷ token-share ratio |
|---|---:|---:|---:|---:|
| counterparty | 17.05% | 64.7% | 66.25% | 3.88× |
| transaction_method | 12.10% | 61.4% | 71.69% | 5.92× |
| processor | 2.01% | 11.4% | 13.10% | 6.51× |
| recurring_flag | 0% | 0% | 0% | n/a |

Reading this: `counterparty` and `transaction_method` both occupy a small
per-token footprint but show up in the *majority* of transactions (typically
one short span per transaction, e.g. a single `"ACH"` or `"POS"` token) — so
even though method takes only 12% of tokens, 72% of transactions have *some*
method tag. `processor` is thin on **both** axes: only 13.1% of transactions
have a processor at all. This is exactly the "naive training under-predicts
it" risk the brief warns about — with an 87%-negative class at the
transaction level, a token classifier optimizing pooled cross-entropy can
reach a deceptively low loss while rarely firing on `PROCESSOR` at all.
**Decision: oversample or upweight transactions containing a `PROCESSOR` span
during training** (Phase 3), and treat processor recall specifically as a
metric to watch in Phase 4/6, not just aggregate F1.

## 4. Inter-annotator agreement — the "big three" finding

There are 5 `annotator_id`s. Pairwise, pooled Cohen's kappa (computed once per
pair over all their shared tokens, not averaged per-id — per-id kappa on an
8-token sequence is often undefined/noisy when a sequence has no tag
variance):

| pair | n ids | n tokens | agreement | kappa |
|---|---:|---:|---:|---:|
| ann_8ac1bf × ann_f2aee8 | 674 | 5,434 | 73.5% | 0.600 |
| ann_78009b × ann_8ac1bf | 652 | 5,210 | 74.1% | 0.605 |
| ann_8ac1bf × ann_928994 | 705 | 5,909 | 74.6% | 0.609 |
| ann_3adc6b × ann_928994 | 661 | 5,192 | 84.8% | 0.790 |
| ann_3adc6b × ann_78009b | 647 | 5,379 | 85.6% | 0.798 |
| ann_3adc6b × ann_f2aee8 | 681 | 5,653 | 85.7% | 0.803 |
| **ann_78009b × ann_928994** | **1,971** | **15,698** | **100.000%** | **1.000** |
| **ann_928994 × ann_f2aee8** | **2,002** | **15,989** | **100.000%** | **1.000** |
| **ann_78009b × ann_f2aee8** | **2,007** | **16,717** | **100.000%** | **1.000** |

The three pairs among `{ann_78009b, ann_928994, ann_f2aee8}` — the
**"big three"** — show **exact, zero-mismatch agreement**, verified
exhaustively (not sampled) across all 5,980 train ids and 603 val ids where
two of them co-annotate (~48k train tokens). This holds even on tokens where
the annotator guidelines explicitly say there's no strict rule (e.g.
`TRANSACTION_METHOD` vs `PROCESSOR`). Independent humans do not converge to
zero disagreement on judgment calls at this scale — so we treat these three
`annotator_id`s as **one deterministic labeling process** logged under three
identities, not three independent raters.

The other two — `ann_3adc6b` and `ann_8ac1bf` — disagree with the big three
(and each other) at rates consistent with genuine independent human
annotation (kappa 0.60–0.82). `ann_8ac1bf` is the weaker of the two: kappa
~0.60–0.62 against everyone, and an **O-rate of 69.4%** vs the big three's
~43–44% baseline (`ann_3adc6b`'s O-rate is 51.3%, in between). Per annotator
guideline #5 ("if unsure, mark O"), a much higher O-rate reads as
"more conservative / less confident," not necessarily "wrong" — but it does
mean `ann_8ac1bf`'s non-`O` labels should be weighted as more informative than
their `O` labels are trustworthy as true negatives.

**Coverage check (why this matters for reconciliation):** every single
train/val id has **at least one big-three annotator** — the pair
`(ann_3adc6b, ann_8ac1bf)` never co-occurs (0/10,000 train ids, 0/1,000 val
ids lack big-three coverage; verified exhaustively in `eda.py::big_three_finding`,
asserted in the script rather than just eyeballed). That means a
gold-equivalent row is available for **100%** of train/val: trivially when
both annotators are big-three (already identical), and by simply preferring
the big-three row when the other annotator is from the minority pair.
**This is the reconciliation strategy Phase 2 will implement** — it dominates
both alternatives from the brief (duplication-as-augmentation would inject the
minority annotators' extra noise into training as if it were signal;
token-level tie-breaking is unnecessary complexity when one side is already a
consistent oracle).

**Confusion matrix** (pooled, symmetric, across all annotator pairs) shows
where the minority annotators' disagreement actually lands:

- `O` ↔ `I-TRANSACTION_METHOD`: 2,760 pooled mismatches (the single largest bucket)
- `O` ↔ `I-BANK_SERVICE_EVENT`: 1,846
- `O` ↔ `I-COUNTERPARTY_NAME`: 429, `O` ↔ `I-PROCESSOR`: 436
- `I-FILLER_WORD` ↔ {`I-COUNTERPARTY_NAME`: 391, `I-TRANSACTION_METHOD`: 431, `I-BANK_SERVICE_EVENT`: 307, `I-PROCESSOR`: 56}
- **Zero** confusion *between* the four scored tags themselves (COUNTERPARTY/METHOD/PROCESSOR/RECURRING never swap with each other)

Disagreement is almost entirely "tag it as something vs. call it `O`/filler,"
not "which specific scored category." That's reassuring for modeling: once a
token is confidently non-`O`/non-filler, which scored class it is tends to be
unambiguous across annotators — the model's harder problem is the
boundary/detection decision, not fine-grained classification among the four
scored types.

## 5. Metadata: bank and transaction_type as auxiliary features

**transaction_type** (75.6% DEBIT / 24.4% CREDIT in train) correlates with
field presence in a way that's plausibly useful as an auxiliary input:

| field | CREDIT presence | DEBIT presence |
|---|---:|---:|
| counterparty | 60.2% | 68.2% |
| transaction_method | 49.8% | **78.7%** |
| processor | **15.4%** | 12.4% |
| recurring_flag | 0% | 0% |

`transaction_method` presence nearly doubles from CREDIT to DEBIT (outgoing
debits are far more likely to specify POS/ACH/CHECK/etc.), and `processor` is
mildly *more* common on CREDIT (incoming payments more often name a named
processor like "SQUARE INC" or "PAYCHEX"). Cheap, worth including.

**bank**: 547 unique banks in train, heavily long-tailed (`DecisionLogic`
alone accounts for 970 rows; most banks have a handful). For banks with ≥10
tokens of a given scored tag, per-bank vocabulary concentration
(`top1_share` = share of that bank's occurrences taken by the single most
common token):

| field | banks meeting threshold | median top1_share | notable example |
|---|---:|---:|---|
| processor | 102 / 547 | 0.40 | `James_Polk_Stone_Community_Bank` → 100% `"bizpay"` |
| transaction_method | 434 / 547 | 0.34 | `Vermont_Federal_Credit_Union` → 100% `"pos"` |
| counterparty | 476 / 547 | 0.12 | (no strong concentration — merchant names are inherently diverse) |

Bank identity is a moderately informative feature for `processor` and
`transaction_method` (many banks show heavy, sometimes total, concentration on
one token/phrasing), but weak for `counterparty`, as expected. Caveat: only
102–476 of 547 banks clear the ≥10-token threshold at all, and the
distribution is dominated by a handful of high-volume banks — most banks in
test will have few or zero matching training examples, so a learned bank
embedding will help disproportionately on the high-volume banks and add
little for the long tail. **Decision:** try prepending `bank` +
`transaction_type` as lightweight auxiliary context (e.g., a special token
prefix) in Phase 6, compared against the baseline on the local harness rather
than assumed to help.

## How this changes the training strategy (summary)

1. **`recurring_flag` is rule-based, full stop** — zero training signal exists; a
   keyword/regex detector is not an optional upgrade, it's the only path. Model
   predictions for this field should be ignored or ensembled defensively, not
   trusted.
2. **Upweight/oversample `PROCESSOR`-bearing transactions** during training — only
   13.1% of transactions have one, and naive pooled cross-entropy risks
   under-predicting it. Track processor recall specifically, not just aggregate F1.
3. **Reconciliation = prefer the big-three annotator's row.** Verified to cover
   100% of train/val ids and to be internally perfectly self-consistent. This
   is simpler and less noisy than duplication-as-augmentation or per-token
   tie-breaking, and Phase 2 implements it as the primary strategy (keeping the
   alternatives' analysis code for comparison, per the brief).
4. **`transaction_type` and `bank` are worth trying as auxiliary features**,
   with realistic expectations that `bank` mostly helps for a small set of
   high-volume banks given the long-tailed distribution.
5. Loss weighting toward the 4 scored tags is worth trying since 69% of tokens
   carry unscored tags that a plain per-token objective would otherwise treat
   as equally important.
