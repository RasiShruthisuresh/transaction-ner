"""Unit tests for the reusable EDA logic (explode_tokens, agreement
math) using a small hand-crafted synthetic frame -- not the live API.
The full EDA in eda.py is exploratory and reviewed by eyeballing
eda_outputs/, but the agreement computation feeds directly into the
Phase 2 reconciliation decision, so its correctness is worth pinning
down with a known-answer test.
"""
from __future__ import annotations

import pandas as pd

from eda import (
    _id_pair_tag_sequences,
    annotator_agreement,
    explode_tokens,
    field_presence_rates,
    pairwise_agreement_summary,
    tag_frequency,
)

TOKENS = ["SQUARE", "INC", "Uplift", "Llc"]


def _row(tx_id, annotator, tags):
    return {
        "id": tx_id,
        "annotator_id": annotator,
        "tokens": TOKENS,
        "ner_tags": tags,
        "bank": "TestBank",
        "transaction_type": "DEBIT",
    }


def _synthetic_df() -> pd.DataFrame:
    # id "a": annotators agree on 3/4 tokens (differ on token 2)
    rows = [
        _row("a", "ann_1", ["I-PROCESSOR", "I-PROCESSOR", "I-COUNTERPARTY_NAME", "O"]),
        _row("a", "ann_2", ["I-PROCESSOR", "I-PROCESSOR", "O", "O"]),
        # id "b": annotators agree on all 4 tokens
        _row("b", "ann_1", ["O", "O", "I-COUNTERPARTY_NAME", "I-COUNTERPARTY_NAME"]),
        _row("b", "ann_2", ["O", "O", "I-COUNTERPARTY_NAME", "I-COUNTERPARTY_NAME"]),
    ]
    return pd.DataFrame(rows)


def test_explode_tokens_row_count_and_fields():
    df = _synthetic_df()
    exploded = explode_tokens(df)
    assert len(exploded) == 4 * 4  # 4 rows x 4 tokens
    assert set(exploded["id"]) == {"a", "b"}
    assert (exploded.loc[exploded["id"] == "a", "bank"] == "TestBank").all()


def test_id_pair_tag_sequences_orders_by_annotator_id():
    df = _synthetic_df()
    pairs = _id_pair_tag_sequences(df)
    assert len(pairs) == 2
    for tx_id, a, b, tags_a, tags_b in pairs:
        assert a < b  # stable ordering
        assert len(tags_a) == len(tags_b) == 4


def test_annotator_agreement_known_values():
    df = _synthetic_df()
    agreement = annotator_agreement(df).set_index("id")
    # id "a": 3 of 4 tokens match (position 2 differs) -> 0.75
    assert agreement.loc["a", "agreement_rate"] == 0.75
    assert agreement.loc["a", "n_matches"] == 3
    # id "b": all 4 match -> 1.0
    assert agreement.loc["b", "agreement_rate"] == 1.0


def test_pairwise_agreement_summary_pools_correctly():
    df = _synthetic_df()
    summary = pairwise_agreement_summary(df)
    assert len(summary) == 1  # only one annotator pair in this synthetic set
    row = summary.iloc[0]
    assert row["n_ids"] == 2
    assert row["n_tokens"] == 8  # 2 ids x 4 tokens
    # 7 of 8 pooled tokens match (only id "a" position 2 differs)
    assert row["agreement_rate"] == 7 / 8


def test_field_presence_rates_matches_hand_count():
    df = _synthetic_df()
    exploded = explode_tokens(df)
    presence = field_presence_rates(df, exploded)
    # I-COUNTERPARTY_NAME: appears in rows for id "a" (ann_1 only) and
    # id "b" (both annotators) -> 3 of 4 rows, 2 of 2 unique ids
    cp = presence.loc["counterparty"]
    assert cp["row_presence_rate"] == 3 / 4
    assert cp["txn_presence_rate"] == 1.0
    # I-PROCESSOR: appears in both annotator rows of id "a" only
    proc = presence.loc["processor"]
    assert proc["row_presence_rate"] == 2 / 4
    assert proc["txn_presence_rate"] == 0.5


def test_tag_frequency_sums_to_total_tokens():
    df = _synthetic_df()
    exploded = explode_tokens(df)
    freq = tag_frequency(exploded)
    assert freq["token_count"].sum() == len(exploded)
    assert abs(freq["token_share"].sum() - 1.0) < 1e-9
