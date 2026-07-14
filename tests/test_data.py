"""Smoke tests for data.py.

These hit the live dataset API (no auth, small payloads), so they're
integration tests rather than pure unit tests -- there's no local
fixture that would exercise the download path meaningfully. Run with:
    .venv/Scripts/python.exe -m pytest tests/test_data.py -v
"""
from __future__ import annotations

import pandas as pd

from config import EXPECTED_SHAPES, TAG_TO_FIELD
from data import load_split, validate_schema


def test_train_shape_and_schema():
    df = load_split("train")
    assert len(df) == EXPECTED_SHAPES["train"]["rows"]
    assert df["id"].nunique() == EXPECTED_SHAPES["train"]["unique_ids"]
    validate_schema(df, "train")  # should not raise


def test_val_shape_and_schema():
    df = load_split("val")
    assert len(df) == EXPECTED_SHAPES["val"]["rows"]
    validate_schema(df, "val")


def test_test_split_has_no_labels():
    df = load_split("test")
    assert len(df) == EXPECTED_SHAPES["test"]["rows"]
    # test.jsonl omits annotator_id and ner_tags entirely
    assert df["ner_tags"].isna().all()
    assert df["annotator_id"].isna().all()
    # every test id must be unique (one row per transaction)
    assert df["id"].is_unique


def test_train_ids_have_two_identical_token_lists():
    df = load_split("train", validate=False)
    sample_ids = df["id"].drop_duplicates().sample(20, random_state=0)
    for tx_id in sample_ids:
        group = df[df["id"] == tx_id]
        assert len(group) == 2
        toks = group["tokens"].tolist()
        assert toks[0] == toks[1], f"tokens differ for id={tx_id}"


def test_tokens_and_tags_same_length():
    df = load_split("train")
    lengths_match = df.apply(lambda r: len(r["tokens"]) == len(r["ner_tags"]), axis=1)
    assert lengths_match.all(), "some rows have tokens/ner_tags length mismatch"


def test_scored_field_names_match_submission_spec():
    # Guards against a typo silently breaking the submission format later.
    assert set(TAG_TO_FIELD.values()) == {
        "counterparty",
        "transaction_method",
        "processor",
        "recurring_flag",
    }
