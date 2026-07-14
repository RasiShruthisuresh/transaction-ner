"""Token-level diagnostics for a trained checkpoint -- complements
eval_harness.py (which scores the 4 *field* strings) with a per-tag
confusion matrix over all 8 tags. Built for Phase 6: Phase 5's baseline
checkpoint showed predicted counterparty text that looked like
BANK_SERVICE_EVENT vocabulary ("withdrawal"/"external"/"deposit"), but
that was a qualitative read of top predicted tokens, not a number. This
module quantifies it: how often does a token whose TRUE tag is X get
PREDICTED as Y, for every (X, Y) pair.
"""
from __future__ import annotations

import argparse
import json
import logging

import pandas as pd
import torch

from config import ALL_TAGS
from data import load_split
from infer import load_checkpoint
from predict import predict_all
from reconcile import reconcile_big_three

logger = logging.getLogger(__name__)


def token_confusion(gold_tags: list[list[str]], pred_tags: list[list[str]]) -> pd.DataFrame:
    """Confusion matrix, rows=true tag, cols=predicted tag, over ALL_TAGS.
    gold_tags/pred_tags are parallel lists of per-transaction tag lists
    (already realigned to one tag per original token)."""
    counts = pd.DataFrame(0, index=ALL_TAGS, columns=ALL_TAGS, dtype=int)
    for g_seq, p_seq in zip(gold_tags, pred_tags):
        assert len(g_seq) == len(p_seq), "gold/pred tag sequences must be the same length"
        for g, p in zip(g_seq, p_seq):
            counts.loc[g, p] += 1
    return counts


def per_tag_precision_recall(confusion: pd.DataFrame) -> pd.DataFrame:
    """Standard per-class precision/recall/F1 read off the confusion
    matrix's diagonal vs row/column sums."""
    rows = []
    for tag in confusion.index:
        tp = confusion.loc[tag, tag]
        support = confusion.loc[tag].sum()
        predicted = confusion[tag].sum()
        precision = tp / predicted if predicted else 0.0
        recall = tp / support if support else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        rows.append({"tag": tag, "support": int(support), "precision": precision, "recall": recall, "f1": f1})
    return pd.DataFrame(rows).set_index("tag")


def diagnose_checkpoint(ckpt_path, batch_size: int = 32, device=None) -> dict:
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, tokenizer, model_config = load_checkpoint(ckpt_path)
    model.to(device)

    val_gold_df = reconcile_big_three(load_split("val"))
    pred_tags = predict_all(model, tokenizer, val_gold_df["tokens"].tolist(), device,
                             batch_size=batch_size, max_length=model_config.max_length)
    gold_tags = val_gold_df["ner_tags"].tolist()

    confusion = token_confusion(gold_tags, pred_tags)
    per_tag = per_tag_precision_recall(confusion)

    cp_as_bse = int(confusion.loc["I-BANK_SERVICE_EVENT", "I-COUNTERPARTY_NAME"])
    bse_support = int(confusion.loc["I-BANK_SERVICE_EVENT"].sum())

    return {
        "confusion": confusion,
        "per_tag": per_tag,
        "bank_service_event_predicted_as_counterparty": cp_as_bse,
        "bank_service_event_support": bse_support,
        "bank_service_event_leak_rate": cp_as_bse / bse_support if bse_support else 0.0,
    }


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--log-file", default=None)
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    result = diagnose_checkpoint(args.ckpt, batch_size=args.batch_size)

    summary = {
        "per_tag": result["per_tag"].round(4).to_dict(orient="index"),
        "bank_service_event_predicted_as_counterparty": result["bank_service_event_predicted_as_counterparty"],
        "bank_service_event_support": result["bank_service_event_support"],
        "bank_service_event_leak_rate": round(result["bank_service_event_leak_rate"], 4),
    }
    text = json.dumps(summary, indent=2)
    if args.log_file:
        with open(args.log_file, "w", encoding="utf-8") as f:
            f.write(text)
            f.write("\n\nfull confusion matrix:\n")
            f.write(result["confusion"].to_string())
        logger.info("wrote diagnostics to %s", args.log_file)
    print(text)
