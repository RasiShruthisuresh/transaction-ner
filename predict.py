"""Inference: run a trained token classifier over transactions and return
one predicted tag per ORIGINAL token (not per subword), by reading off
the prediction at each token's first-subword position -- symmetric with
how tokenization.align_labels assigns labels during training. Used by
train.py (val-loss-only during training doesn't need this, but Phase 4
eval harness does) and infer.py (Phase 7, full test-set inference).
"""
from __future__ import annotations

import torch

from config import ID2TAG
from tokenization import normalize_case


@torch.no_grad()
def predict_tags(model, tokenizer, tokens_batch: list[list[str]], device, max_length: int = 64) -> list[list[str]]:
    """tokens_batch: list of pre-tokenized transactions (list[str] each).
    Returns: list of predicted tag lists, one per input transaction, each
    the same length as the corresponding input token list.
    """
    model.eval()
    enc = tokenizer(
        [normalize_case(tokens) for tokens in tokens_batch],
        is_split_into_words=True,
        truncation=True,
        max_length=max_length,
        padding=True,
        return_tensors="pt",
    )
    input_ids = enc["input_ids"].to(device)
    attention_mask = enc["attention_mask"].to(device)
    logits = model(input_ids=input_ids, attention_mask=attention_mask)["logits"]
    pred_ids = logits.argmax(-1).cpu().tolist()

    results = []
    for i, tokens in enumerate(tokens_batch):
        word_ids = enc.word_ids(batch_index=i)
        tags = ["O"] * len(tokens)
        seen = set()
        for pos, wid in enumerate(word_ids):
            if wid is not None and wid not in seen:
                seen.add(wid)
                tags[wid] = ID2TAG[pred_ids[i][pos]]
        results.append(tags)
    return results


def predict_all(model, tokenizer, tokens_lists: list[list[str]], device, batch_size: int = 32, max_length: int = 64) -> list[list[str]]:
    """Batched wrapper over predict_tags for large lists (e.g. the 10k-row
    test set in Phase 7)."""
    out = []
    for start in range(0, len(tokens_lists), batch_size):
        batch = tokens_lists[start : start + batch_size]
        out.extend(predict_tags(model, tokenizer, batch, device, max_length=max_length))
    return out
