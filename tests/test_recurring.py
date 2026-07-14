"""Unit tests for the rule-based recurring_flag detector (recurring.py).

Pure regex logic, no model needed -- see EDA_FINDINGS.md sec 2 for why
this field is rule-based rather than learned.
"""
from __future__ import annotations

import pytest

from recurring import detect_recurring


@pytest.mark.parametrize(
    "tokens,expected_hit",
    [
        (["POS", "WITHDRAWAL", "RECURRING", "PAYPAL"], "RECURRING"),
        (["ACH", "PREAUTHORIZED", "DEBIT"], "PREAUTHORIZED"),
        (["NETFLIX", "SUBSCRIPTION", "FEE"], "SUBSCRIPTION"),
        (["AUTOPAY", "UTILITY", "BILL"], "AUTOPAY"),
        (["AUTODEBIT", "INSURANCE"], "AUTODEBIT"),
        (["AUTORENEW", "MEMBERSHIP"], "AUTORENEW"),
        (["AUTOAPPROVED", "TXN"], "AUTOAPPROVED"),
        (["recur", "lowercase", "match"], "recur"),  # case-insensitive
    ],
)
def test_detect_recurring_fires_on_known_keyword_families(tokens, expected_hit):
    assert detect_recurring(tokens) == expected_hit


def test_detect_recurring_empty_when_no_keyword_present():
    assert detect_recurring(["POS", "WITHDRAWAL", "PAYPAL", "INST", "XFER"]) == ""


def test_detect_recurring_does_not_fire_on_unrelated_auto_words():
    # "auto" alone (not auto+debit/pay/approv/renew) must not false-positive,
    # e.g. "AUTOMATIC" / "AUTOMOBILE" style tokens.
    assert detect_recurring(["AUTOMATIC", "PAYMENT", "SYSTEM"]) == ""
    assert detect_recurring(["AUTOMOBILE", "LOAN"]) == ""


def test_detect_recurring_joins_multiple_hits_in_token_order():
    assert detect_recurring(["PREAUTH", "RECURRING", "CHARGE"]) == "PREAUTH RECURRING"


def test_detect_recurring_empty_tokens_list():
    assert detect_recurring([]) == ""
