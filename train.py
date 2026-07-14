"""Baseline training loop: fine-tune a token classifier on the reconciled
train set, tracking loss on the reconciled val set each epoch.

Config is CLI args, not hardcoded, so Phase 6 upgrades (different
encoder, different hyperparams, subsampled smoke tests) are a rerun with
different flags, not a code change. Plain PyTorch loop rather than HF
Trainer -- fewer moving parts to debug on a first end-to-end run, and
precise control over log-file redirection (see CLAUDE.md: training
stdout must go to logs/, not the terminal).
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader

from config import CHECKPOINT_DIR, SEED, set_seed
from data import load_split
from model import ModelConfig, build_model, build_tokenizer
from reconcile import reconcile_big_three
from tokenization import TransactionDataset

logger = logging.getLogger(__name__)


def parse_args(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--model-name", default="jhu-clsp/ettin-encoder-32m")
    p.add_argument("--max-length", type=int, default=64)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=3e-5)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--max-train-examples", type=int, default=None, help="subsample train for smoke tests")
    p.add_argument("--max-val-examples", type=int, default=None, help="subsample val for smoke tests")
    p.add_argument("--run-name", default="baseline")
    p.add_argument("--log-file", default=None, help="e.g. logs/phase3_train.log")
    return p.parse_args(argv)


def setup_logging(log_file: str | None) -> None:
    handlers = [logging.StreamHandler(sys.stdout)]
    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file, mode="w"))
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", handlers=handlers, force=True)


def run_epoch(model, loader, optimizer, device, train: bool) -> float:
    model.train(train)
    total_loss, n_batches = 0.0, 0
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.set_grad_enabled(train):
            out = model(**batch)
            loss = out["loss"]
        if train:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        total_loss += loss.item()
        n_batches += 1
    return total_loss / max(n_batches, 1)


def load_gold(split: str, max_examples: int | None):
    raw = load_split(split)
    gold = reconcile_big_three(raw)
    if max_examples is not None and max_examples < len(gold):
        gold = gold.sample(n=max_examples, random_state=SEED).reset_index(drop=True)
    return gold


def main(argv=None):
    args = parse_args(argv)
    set_seed(SEED)
    setup_logging(args.log_file)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("run=%s device=%s model=%s epochs=%d batch_size=%d lr=%g",
                args.run_name, device, args.model_name, args.epochs, args.batch_size, args.lr)

    train_gold = load_gold("train", args.max_train_examples)
    val_gold = load_gold("val", args.max_val_examples)
    logger.info("train_examples=%d val_examples=%d", len(train_gold), len(val_gold))

    model_config = ModelConfig(model_name=args.model_name, max_length=args.max_length)
    tokenizer = build_tokenizer(model_config)
    model = build_model(model_config).to(device)
    logger.info("model params=%d", sum(p.numel() for p in model.parameters()))

    train_ds = TransactionDataset(train_gold, tokenizer, max_length=args.max_length, has_labels=True)
    val_ds = TransactionDataset(val_gold, tokenizer, max_length=args.max_length, has_labels=True)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    optimizer = AdamW(model.parameters(), lr=args.lr)

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    ckpt_path = CHECKPOINT_DIR / f"{args.run_name}.pt"
    best_val_loss = float("inf")

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_loss = run_epoch(model, train_loader, optimizer, device, train=True)
        val_loss = run_epoch(model, val_loader, optimizer, device, train=False)
        logger.info("epoch=%d train_loss=%.4f val_loss=%.4f elapsed=%.1fs",
                     epoch, train_loss, val_loss, time.time() - t0)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({"model_state": model.state_dict(), "config": model_config}, ckpt_path)
            logger.info("saved checkpoint to %s (val_loss=%.4f)", ckpt_path, val_loss)

    logger.info("done. best_val_loss=%.4f ckpt=%s", best_val_loss, ckpt_path)
    return {"best_val_loss": best_val_loss, "ckpt_path": str(ckpt_path)}


if __name__ == "__main__":
    main()
