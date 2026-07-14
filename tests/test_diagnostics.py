"""Unit tests for diagnostics.py's token-level confusion matrix -- pure
logic on hand-built tag sequences, no model needed."""
from __future__ import annotations

from diagnostics import per_tag_precision_recall, token_confusion

GOLD = [
    ["I-COUNTERPARTY_NAME", "O", "I-BANK_SERVICE_EVENT"],
    ["I-BANK_SERVICE_EVENT", "I-BANK_SERVICE_EVENT"],
]
# transaction 1: counterparty correct, O correct, BSE mispredicted as counterparty
# transaction 2: one BSE correct, one BSE mispredicted as counterparty
PRED = [
    ["I-COUNTERPARTY_NAME", "O", "I-COUNTERPARTY_NAME"],
    ["I-BANK_SERVICE_EVENT", "I-COUNTERPARTY_NAME"],
]


def test_token_confusion_hand_computed_cells():
    confusion = token_confusion(GOLD, PRED)
    assert confusion.loc["I-COUNTERPARTY_NAME", "I-COUNTERPARTY_NAME"] == 1
    assert confusion.loc["O", "O"] == 1
    assert confusion.loc["I-BANK_SERVICE_EVENT", "I-COUNTERPARTY_NAME"] == 2
    assert confusion.loc["I-BANK_SERVICE_EVENT", "I-BANK_SERVICE_EVENT"] == 1
    assert confusion.values.sum() == 5  # 3 + 2 tokens total


def test_per_tag_precision_recall_hand_computed():
    confusion = token_confusion(GOLD, PRED)
    per_tag = per_tag_precision_recall(confusion)
    # I-COUNTERPARTY_NAME: predicted 3 times (1 correct + 2 leaked from BSE), true support 1
    assert per_tag.loc["I-COUNTERPARTY_NAME", "support"] == 1
    assert per_tag.loc["I-COUNTERPARTY_NAME", "precision"] == 1 / 3
    assert per_tag.loc["I-COUNTERPARTY_NAME", "recall"] == 1.0
    # I-BANK_SERVICE_EVENT: true support 3, predicted correctly only once
    assert per_tag.loc["I-BANK_SERVICE_EVENT", "support"] == 3
    assert per_tag.loc["I-BANK_SERVICE_EVENT", "recall"] == 1 / 3
    assert per_tag.loc["I-BANK_SERVICE_EVENT", "precision"] == 1.0


def test_token_confusion_requires_matching_lengths():
    import pytest

    with pytest.raises(AssertionError):
        token_confusion([["O", "O"]], [["O"]])
