"""Unit tests for reconcile.py: synthetic well-formedness checks plus a
real-data run against train/val to confirm the shape/id/token guarantees
hold outside the synthetic case."""
from __future__ import annotations

import pandas as pd
import pytest

from config import BIG_THREE_ANNOTATORS
from data import load_split
from reconcile import (
    reconcile_big_three,
    reconcile_duplication,
    reconcile_token_tiebreak,
    validate_reconciled,
)

TOKENS = ["SQUARE", "INC", "Recurring", "Llc"]
BIG3 = next(iter(BIG_THREE_ANNOTATORS))  # any big-three id
MINORITY = "ann_3adc6b"


def _row(tx_id, annotator, tags):
    return {"id": tx_id, "annotator_id": annotator, "tokens": TOKENS, "ner_tags": tags,
            "bank": "TestBank", "transaction_type": "DEBIT"}


def _synthetic_df() -> pd.DataFrame:
    rows = [
        # id "a": big-three says O on token 2, minority says a real tag there
        _row("a", BIG3, ["I-PROCESSOR", "I-PROCESSOR", "O", "O"]),
        _row("a", MINORITY, ["I-PROCESSOR", "I-PROCESSOR", "I-RECURRING_FLAG", "O"]),
        # id "b": both non-O and disagree on token 2 (contrived, tests the big-three fallback)
        _row("b", BIG3, ["O", "O", "I-COUNTERPARTY_NAME", "O"]),
        _row("b", MINORITY, ["O", "O", "I-FILLER_WORD", "O"]),
    ]
    return pd.DataFrame(rows)


def test_reconcile_big_three_picks_the_big_three_row():
    df = _synthetic_df()
    gold = reconcile_big_three(df)
    assert len(gold) == 2
    assert set(gold["annotator_id"]) == {BIG3}
    row_a = gold[gold["id"] == "a"].iloc[0]
    assert row_a["ner_tags"] == ["I-PROCESSOR", "I-PROCESSOR", "O", "O"]


def test_reconcile_big_three_raises_if_no_big_three_annotator():
    df = pd.DataFrame([
        _row("c", MINORITY, ["O", "O", "O", "O"]),
        _row("c", "ann_8ac1bf", ["O", "O", "O", "O"]),
    ])
    with pytest.raises(AssertionError, match="no big-three annotator"):
        reconcile_big_three(df)


def test_reconcile_duplication_is_passthrough():
    df = _synthetic_df()
    out = reconcile_duplication(df)
    assert len(out) == len(df)
    assert set(out["annotator_id"]) == {BIG3, MINORITY}


def test_reconcile_token_tiebreak_prefers_nonO_and_big_three_fallback():
    df = _synthetic_df()
    out = reconcile_token_tiebreak(df).set_index("id")
    # id "a": token 2 -- big-three O, minority non-O -> take minority's tag
    assert out.loc["a", "ner_tags"] == ["I-PROCESSOR", "I-PROCESSOR", "I-RECURRING_FLAG", "O"]
    # id "b": token 2 -- both non-O, disagree -> fall back to big-three's tag
    assert out.loc["b", "ner_tags"] == ["O", "O", "I-COUNTERPARTY_NAME", "O"]


def test_validate_reconciled_passes_on_well_formed_frame():
    df = _synthetic_df()
    gold = reconcile_big_three(df)
    validate_reconciled(gold, df, expected_rows=2)  # should not raise


def test_validate_reconciled_catches_wrong_row_count():
    df = _synthetic_df()
    gold = reconcile_big_three(df)
    with pytest.raises(AssertionError, match="expected"):
        validate_reconciled(gold, df, expected_rows=99)


def test_real_train_val_reconciliation_is_well_formed():
    for split, expected in (("train", 10000), ("val", 1000)):
        raw = load_split(split)
        gold = reconcile_big_three(raw)
        validate_reconciled(gold, raw, expected_rows=expected)
        assert set(gold["annotator_id"]) <= BIG_THREE_ANNOTATORS
