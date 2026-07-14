"""Shared tokenization + subword-label alignment for token classification.

Used by train.py (labels present) and predict.py (labels absent). Kept
separate from model.py (architecture) and train.py (training loop)
because subword alignment is the most bug-prone part of adapting
pre-tokenized NER data to a subword tokenizer, and both training and
inference need the exact same alignment convention to stay consistent.
"""
from __future__ import annotations

import torch
from torch.utils.data import Dataset

from config import TAG2ID


def align_labels(word_ids: list[int | None], tags: list[str]) -> list[int]:
    """Map a tokenizer's word_ids() (one entry per subword position, or
    None for special tokens like [CLS]/[SEP]/padding) to label ids.

    Design decision: only the FIRST subword of each original token
    carries the label; every other subword of that token (and all
    special/padding tokens) gets -100, PyTorch CrossEntropyLoss's
    ignore_index. We deliberately do NOT repeat the label across every
    subword of a token: that would inflate the loss weight of tokens
    that happen to split into more subpieces for reasons unrelated to
    their label (e.g. rare merchant names), and at inference we only
    need one label per ORIGINAL token to reconstruct the submission
    string, so predicting from the first subword only is sufficient.
    """
    labels = []
    prev_word_id = None
    for wid in word_ids:
        if wid is None or wid == prev_word_id:
            labels.append(-100)
        else:
            labels.append(TAG2ID[tags[wid]])
        prev_word_id = wid
    return labels


class TransactionDataset(Dataset):
    """One example per transaction row. Expects a DataFrame with `tokens`
    (list[str]) and, if has_labels, `ner_tags` (list[str]) columns --
    i.e. the output of reconcile.reconcile_big_three() for train/val, or
    data.load_split("test") as-is (has_labels=False)."""

    def __init__(self, df, tokenizer, max_length: int = 64, has_labels: bool = True):
        self.tokens = df["tokens"].tolist()
        self.tags = df["ner_tags"].tolist() if has_labels else None
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.has_labels = has_labels

    def __len__(self):
        return len(self.tokens)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            self.tokens[idx],
            is_split_into_words=True,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
        )
        item = {
            "input_ids": torch.tensor(enc["input_ids"]),
            "attention_mask": torch.tensor(enc["attention_mask"]),
        }
        if self.has_labels:
            word_ids = enc.word_ids()
            item["labels"] = torch.tensor(align_labels(word_ids, self.tags[idx]))
        return item
