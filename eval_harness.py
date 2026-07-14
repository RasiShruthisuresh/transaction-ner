"""Local evaluation harness mirroring the server's macro-F1-across-4-fields
metric as closely as possible from the brief's description.

CALIBRATED (Phase 5, first real submission, 2026-07-14): the server's
response reports each of the 3 bag-of-words fields with
`"metric": "token"` plus precision/recall/f1/support, e.g. counterparty
precision=0.4237 recall=0.2228 support=6581. Back-solving
(TP = recall*support, FP = TP/precision - TP) reproduces the reported F1
exactly for every field, confirming these are POOLED precision/recall/F1
over token counts summed across the whole test set -- structurally the
same computation as `field_score_micro` below, not an average of
per-transaction F1s. **`final_score_micro` is therefore the calibrated
local proxy; `final_score_macro` is kept only as a documented
alternative**, not because it's expected to match, but because it was
already built and costs nothing to keep computing.

The server's own headline number ("this_submission_macro_f1", despite
the name) is just the plain mean of the 4 per-field F1s -- confirmed by
recomputing it from the 4 reported field F1s in that same response
((0.292+0.5204+0.2623+0.9231)/4 = 0.4994, matches exactly). That's the
same shape as `final_score_micro`/`final_score_macro` below, so no
change needed there beyond picking micro as the per-field aggregation.

recurring_flag's presence-based F1 has no micro/macro ambiguity in the
same sense (it's one binary decision per transaction, not a multiset
comparison) -- confirmed by the same submission, which reports it with
`"metric": "presence"`, matching `presence_score` below exactly.

Still uncalibrated: the empty-vs-empty convention for bag-of-words F1
(treated here as a perfect match, P=R=F1=1.0) can't be distinguished
from an "undefined/skipped" convention using the aggregate numbers alone.
Low priority -- doesn't explain any of the gap seen in the first
submission, see PROGRESS.md Phase 6/7 for what did (a case-sensitivity
bug, not a metric mismatch).

See EDA_FINDINGS.md sec 2: val has ZERO positive recurring_flag labels,
so recall for that field can never be validated locally against val --
only against a real submission (Phase 5).
"""
from __future__ import annotations

from collections import Counter

from config import SCORED_FIELDS

BOW_FIELDS = [f for f in SCORED_FIELDS if f != "recurring_flag"]


def _normalize(field_string: str) -> list[str]:
    """Case- and whitespace-insensitive tokenization of a submission
    field string, per the brief's bag-of-words scoring description."""
    return field_string.casefold().split()


def bow_prf1(pred: str, gold: str) -> tuple[float, float, float]:
    """Bag-of-words precision/recall/F1 for one field on one transaction."""
    pred_toks = Counter(_normalize(pred))
    gold_toks = Counter(_normalize(gold))
    overlap = sum((pred_toks & gold_toks).values())
    pred_total = sum(pred_toks.values())
    gold_total = sum(gold_toks.values())

    if pred_total == 0 and gold_total == 0:
        return 1.0, 1.0, 1.0
    precision = overlap / pred_total if pred_total else 0.0
    recall = overlap / gold_total if gold_total else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return precision, recall, f1


def field_score_micro(preds: list[str], golds: list[str]) -> dict:
    """Pool overlap/pred_total/gold_total counts across all examples,
    then compute one P/R/F1 from the pooled counts (large transactions
    dominate the aggregate under this scheme)."""
    overlap = pred_total = gold_total = 0
    for pred, gold in zip(preds, golds):
        pred_toks = Counter(_normalize(pred))
        gold_toks = Counter(_normalize(gold))
        overlap += sum((pred_toks & gold_toks).values())
        pred_total += sum(pred_toks.values())
        gold_total += sum(gold_toks.values())
    precision = overlap / pred_total if pred_total else (1.0 if gold_total == 0 else 0.0)
    recall = overlap / gold_total if gold_total else (1.0 if pred_total == 0 else 0.0)
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}


def field_score_macro(preds: list[str], golds: list[str]) -> dict:
    """Mean of per-example bag-of-words P/R/F1 across all examples (every
    transaction weighted equally regardless of field length)."""
    scores = [bow_prf1(p, g) for p, g in zip(preds, golds)]
    n = len(scores)
    return {
        "precision": sum(s[0] for s in scores) / n,
        "recall": sum(s[1] for s in scores) / n,
        "f1": sum(s[2] for s in scores) / n,
    }


def presence_score(preds: list[str], golds: list[str]) -> dict:
    """Standard binary P/R/F1 for recurring_flag: non-empty prediction vs
    non-empty gold, pooled TP/FP/FN/TN across all transactions."""
    tp = fp = fn = tn = 0
    for pred, gold in zip(preds, golds):
        p_present, g_present = bool(pred.strip()), bool(gold.strip())
        if p_present and g_present:
            tp += 1
        elif p_present and not g_present:
            fp += 1
        elif not p_present and g_present:
            fn += 1
        else:
            tn += 1
    precision = tp / (tp + fp) if (tp + fp) else (1.0 if (tp + fn) == 0 else 0.0)
    recall = tp / (tp + fn) if (tp + fn) else (1.0 if (tp + fp) == 0 else 0.0)
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return {"precision": precision, "recall": recall, "f1": f1, "tp": tp, "fp": fp, "fn": fn, "tn": tn}


def score_predictions(gold_fields: dict[str, dict[str, str]], pred_fields: dict[str, dict[str, str]]) -> dict:
    """gold_fields / pred_fields: {id: {field: string}}.

    Returns a report with micro+macro P/R/F1 for the 3 bag-of-words
    fields, presence P/R/F1 for recurring_flag, and both possible
    final-score aggregations (plain average of the 4 field F1s, once
    using each field's micro F1 and once using macro F1).
    """
    ids = sorted(set(gold_fields) & set(pred_fields))
    missing = set(gold_fields) - set(pred_fields)
    assert not missing, f"{len(missing)} gold ids missing from predictions: {list(missing)[:5]}"

    report: dict = {}
    for field in BOW_FIELDS:
        preds = [pred_fields[i][field] for i in ids]
        golds = [gold_fields[i][field] for i in ids]
        report[field] = {"micro": field_score_micro(preds, golds), "macro": field_score_macro(preds, golds)}

    rec_preds = [pred_fields[i]["recurring_flag"] for i in ids]
    rec_golds = [gold_fields[i]["recurring_flag"] for i in ids]
    report["recurring_flag"] = presence_score(rec_preds, rec_golds)

    report["final_score_micro"] = (sum(report[f]["micro"]["f1"] for f in BOW_FIELDS) + report["recurring_flag"]["f1"]) / 4
    report["final_score_macro"] = (sum(report[f]["macro"]["f1"] for f in BOW_FIELDS) + report["recurring_flag"]["f1"]) / 4
    report["final_score"] = report["final_score_micro"]  # calibrated choice, see module docstring
    report["n_examples"] = len(ids)
    return report
