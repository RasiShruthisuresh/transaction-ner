"""Phase 4 test: a hand-constructed synthetic example with independently
computed correct precision/recall/F1, asserted exactly against
eval_harness.py. This is the phase's actual test, not just "it runs" --
see PROGRESS.md / the original brief for why.

Synthetic "counterparty" transactions (worked by hand in comments):
  tx1: gold="SQUARE INC"        pred="SQUARE INC"        -> exact match, f1=1.0
  tx2: gold="UPLIFT OUTDOOR LLC" pred="UPLIFT OUTDOOR"    -> overlap=2, gold_total=3, pred_total=2
       precision=2/2=1.0, recall=2/3, f1=2*1*(2/3)/(1+2/3)=4/5=0.8
  tx3: gold=""                   pred="PAYCHEX"           -> false positive, f1=0.0
  tx4: gold="ADP"                pred=""                  -> false negative, f1=0.0

  macro_f1 = (1.0 + 0.8 + 0.0 + 0.0) / 4 = 0.45
  micro: overlap=2+2+0+0=4, pred_total=2+2+1+0=5, gold_total=2+3+0+1=6
         precision=4/5=0.8, recall=4/6=2/3, f1=2*0.8*(2/3)/(0.8+2/3)=16/22=8/11

Synthetic "recurring_flag" presence (4 transactions, one of each outcome):
  TP, FN, FP, TN -> precision=0.5, recall=0.5, f1=0.5
"""
from __future__ import annotations

import pytest

from eval_harness import bow_prf1, field_score_macro, field_score_micro, presence_score, score_predictions
from fields import extract_fields

GOLD_CP = ["SQUARE INC", "UPLIFT OUTDOOR LLC", "", "ADP"]
PRED_CP = ["SQUARE INC", "UPLIFT OUTDOOR", "PAYCHEX", ""]


def test_bow_prf1_per_example_hand_computed():
    assert bow_prf1(PRED_CP[0], GOLD_CP[0]) == (1.0, 1.0, 1.0)
    p, r, f1 = bow_prf1(PRED_CP[1], GOLD_CP[1])
    assert p == pytest.approx(1.0)
    assert r == pytest.approx(2 / 3)
    assert f1 == pytest.approx(0.8)
    assert bow_prf1(PRED_CP[2], GOLD_CP[2]) == (0.0, 0.0, 0.0)
    assert bow_prf1(PRED_CP[3], GOLD_CP[3]) == (0.0, 0.0, 0.0)


def test_bow_prf1_both_empty_is_perfect():
    assert bow_prf1("", "") == (1.0, 1.0, 1.0)


def test_field_score_macro_hand_computed():
    result = field_score_macro(PRED_CP, GOLD_CP)
    assert result["f1"] == pytest.approx(0.45)


def test_field_score_micro_hand_computed():
    result = field_score_micro(PRED_CP, GOLD_CP)
    assert result["precision"] == pytest.approx(0.8)
    assert result["recall"] == pytest.approx(2 / 3)
    assert result["f1"] == pytest.approx(8 / 11)


def test_presence_score_hand_computed():
    # tx1: TP, tx2: FN, tx3: FP, tx4: TN
    preds = ["AUTO PAY", "", "PREAUTH", ""]
    golds = ["RECURRING", "SUBSCRIPTION", "", ""]
    result = presence_score(preds, golds)
    assert result == {"precision": 0.5, "recall": 0.5, "f1": 0.5, "tp": 1, "fp": 1, "fn": 1, "tn": 1}


def test_score_predictions_end_to_end_matches_hand_computed_values():
    ids = ["tx1", "tx2", "tx3", "tx4"]
    gold_fields = {i: {"counterparty": g, "transaction_method": "", "processor": "", "recurring_flag": ""} for i, g in zip(ids, GOLD_CP)}
    pred_fields = {i: {"counterparty": p, "transaction_method": "", "processor": "", "recurring_flag": ""} for i, p in zip(ids, PRED_CP)}
    report = score_predictions(gold_fields, pred_fields)
    assert report["counterparty"]["macro"]["f1"] == pytest.approx(0.45)
    assert report["counterparty"]["micro"]["f1"] == pytest.approx(8 / 11)
    # transaction_method/processor/recurring_flag all empty-vs-empty -> perfect
    assert report["transaction_method"]["macro"]["f1"] == pytest.approx(1.0)
    assert report["recurring_flag"]["f1"] == pytest.approx(1.0)
    # final_score = mean of the 4 field F1s
    expected_macro_final = (0.45 + 1.0 + 1.0 + 1.0) / 4
    assert report["final_score_macro"] == pytest.approx(expected_macro_final)


def test_score_predictions_raises_on_missing_ids():
    gold_fields = {"a": {"counterparty": "X", "transaction_method": "", "processor": "", "recurring_flag": ""}}
    pred_fields = {}
    with pytest.raises(AssertionError, match="missing from predictions"):
        score_predictions(gold_fields, pred_fields)


def test_extract_fields_multiple_noncontiguous_spans_concatenate_in_order():
    tokens = ["SQUARE", "FOO", "PAYPAL", "BAR"]
    tags = ["I-PROCESSOR", "O", "I-PROCESSOR", "O"]
    fields = extract_fields(tokens, tags)
    assert fields["processor"] == "SQUARE PAYPAL"
    assert fields["counterparty"] == ""


def test_extract_fields_empty_when_no_scored_tags_present():
    fields = extract_fields(["FOO", "BAR"], ["O", "I-FILLER_WORD"])
    assert all(v == "" for v in fields.values())
