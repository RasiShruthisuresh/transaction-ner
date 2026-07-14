"""Unit tests for the subword-label alignment logic in tokenization.py --
the most bug-prone part of adapting pre-tokenized NER data to a subword
tokenizer, so it gets a dedicated, fast, no-training-required test.
"""
from __future__ import annotations

import pandas as pd

from config import TAG2ID
from model import ModelConfig, build_tokenizer
from tokenization import TransactionDataset, align_labels


def test_align_labels_pure_logic():
    # word_ids as a tokenizer would emit them for 3 words where word 1
    # ("INC") splits into 2 subwords: [CLS] SQ ##UARE IN ##C LLC [SEP]
    word_ids = [None, 0, 0, 1, 1, 2, None]
    tags = ["I-PROCESSOR", "I-PROCESSOR", "O"]
    labels = align_labels(word_ids, tags)
    assert labels == [
        -100,  # [CLS]
        TAG2ID["I-PROCESSOR"],  # first subword of word 0
        -100,  # continuation subword of word 0
        TAG2ID["I-PROCESSOR"],  # first subword of word 1
        -100,  # continuation subword of word 1
        TAG2ID["O"],  # first (only) subword of word 2
        -100,  # [SEP]
    ]


def test_align_labels_single_subword_per_word():
    word_ids = [None, 0, 1, 2, None]
    tags = ["O", "I-COUNTERPARTY_NAME", "I-COUNTERPARTY_NAME"]
    labels = align_labels(word_ids, tags)
    assert labels == [-100, TAG2ID["O"], TAG2ID["I-COUNTERPARTY_NAME"], TAG2ID["I-COUNTERPARTY_NAME"], -100]


def test_transaction_dataset_shapes_and_roundtrip():
    tokenizer = build_tokenizer(ModelConfig())
    df = pd.DataFrame(
        [
            {"tokens": ["SQUARE", "INC"], "ner_tags": ["I-PROCESSOR", "I-PROCESSOR"]},
            {"tokens": ["CHECK", "#", "123"], "ner_tags": ["I-TRANSACTION_METHOD", "I-SEPARATOR_PUNCTUATION", "O"]},
        ]
    )
    ds = TransactionDataset(df, tokenizer, max_length=16, has_labels=True)
    assert len(ds) == 2
    item = ds[0]
    assert item["input_ids"].shape == (16,)
    assert item["attention_mask"].shape == (16,)
    assert item["labels"].shape == (16,)
    # exactly 2 non-ignored label positions (one per original token)
    assert (item["labels"] != -100).sum().item() == 2
    non_ignored = item["labels"][item["labels"] != -100].tolist()
    assert non_ignored == [TAG2ID["I-PROCESSOR"], TAG2ID["I-PROCESSOR"]]


def test_transaction_dataset_without_labels():
    tokenizer = build_tokenizer(ModelConfig())
    df = pd.DataFrame([{"tokens": ["FOO", "BAR"]}])
    ds = TransactionDataset(df, tokenizer, max_length=8, has_labels=False)
    item = ds[0]
    assert "labels" not in item
