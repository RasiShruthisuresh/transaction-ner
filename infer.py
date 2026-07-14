"""Inference + submission formatting. Used by Phase 5 (score a baseline
checkpoint on val for calibration, build a first predictions.json) and
Phase 7 (same code, final model)."""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import torch

from config import SCORED_FIELDS
from data import load_split
from eval_harness import score_predictions
from fields import extract_fields
from model import build_model, build_tokenizer
from predict import predict_all
from reconcile import reconcile_big_three

logger = logging.getLogger(__name__)


def load_checkpoint(ckpt_path):
    ckpt = torch.load(ckpt_path, weights_only=False)
    model_config = ckpt["config"]
    model = build_model(model_config)
    model.load_state_dict(ckpt["model_state"])
    tokenizer = build_tokenizer(model_config)
    return model, tokenizer, model_config


def predict_fields_for_df(model, tokenizer, df, model_config, device, batch_size: int = 32) -> dict[str, dict[str, str]]:
    tags = predict_all(model, tokenizer, df["tokens"].tolist(), device, batch_size=batch_size, max_length=model_config.max_length)
    return {tx_id: extract_fields(toks, t) for tx_id, toks, t in zip(df["id"], df["tokens"], tags)}


def score_on_val(ckpt_path, batch_size: int = 32, device=None) -> dict:
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, tokenizer, model_config = load_checkpoint(ckpt_path)
    model.to(device)
    val_gold_df = reconcile_big_three(load_split("val"))
    pred_fields = predict_fields_for_df(model, tokenizer, val_gold_df, model_config, device, batch_size)
    gold_fields = {row.id: extract_fields(row.tokens, row.ner_tags) for row in val_gold_df.itertuples()}
    return score_predictions(gold_fields, pred_fields)


def validate_predictions(records: list[dict], test_df) -> None:
    """All 10,000 test ids present, matching test.jsonl exactly, all four
    scored keys present, no nulls, all string-typed -- checked before a
    predictions.json is ever considered submission-ready."""
    ids_in_records = [r["id"] for r in records]
    assert len(ids_in_records) == len(set(ids_in_records)), "duplicate ids in predictions"
    assert set(ids_in_records) == set(test_df["id"]), "predictions ids must match test.jsonl ids exactly"
    for r in records:
        for field in SCORED_FIELDS:
            assert field in r, f"record {r['id']} missing field {field}"
            assert r[field] is not None, f"record {r['id']} has null {field}"
            assert isinstance(r[field], str), f"record {r['id']} field {field} is not a string"


def build_predictions_json(ckpt_path, out_path: str = "predictions.json", batch_size: int = 32, device=None) -> list[dict]:
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, tokenizer, model_config = load_checkpoint(ckpt_path)
    model.to(device)
    test_df = load_split("test")
    pred_fields = predict_fields_for_df(model, tokenizer, test_df, model_config, device, batch_size)

    records = [{"id": tx_id, **pred_fields[tx_id]} for tx_id in test_df["id"]]
    validate_predictions(records, test_df)

    Path(out_path).write_text(json.dumps(records, indent=2), encoding="utf-8")
    logger.info("wrote %d validated records to %s", len(records), out_path)
    return records


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--out", default="predictions.json")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--score-val", action="store_true", help="score checkpoint on val instead of building predictions.json")
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    if args.score_val:
        report = score_on_val(args.ckpt, batch_size=args.batch_size)
        print(json.dumps(report, indent=2))
    else:
        build_predictions_json(args.ckpt, out_path=args.out, batch_size=args.batch_size)
