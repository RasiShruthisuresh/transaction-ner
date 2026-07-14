"""Unit tests for the submission-format validation logic in infer.py --
pure logic, no model/checkpoint needed, so this runs fast and catches
format regressions before any real inference run."""
from __future__ import annotations

import pandas as pd
import pytest

from infer import validate_predictions

TEST_DF = pd.DataFrame({"id": ["a", "b", "c"]})


def _record(tx_id, **overrides):
    base = {"id": tx_id, "counterparty": "", "transaction_method": "", "processor": "", "recurring_flag": ""}
    base.update(overrides)
    return base


def test_validate_predictions_passes_on_well_formed_records():
    records = [_record("a"), _record("b"), _record("c", counterparty="SQUARE")]
    validate_predictions(records, TEST_DF)  # should not raise


def test_validate_predictions_catches_missing_id():
    records = [_record("a"), _record("b")]  # missing "c"
    with pytest.raises(AssertionError, match="match test.jsonl"):
        validate_predictions(records, TEST_DF)


def test_validate_predictions_catches_duplicate_id():
    records = [_record("a"), _record("a"), _record("b"), _record("c")]
    with pytest.raises(AssertionError, match="duplicate"):
        validate_predictions(records, TEST_DF)


def test_validate_predictions_catches_missing_field():
    records = [_record("a"), _record("b"), _record("c")]
    del records[0]["processor"]
    with pytest.raises(AssertionError, match="missing field"):
        validate_predictions(records, TEST_DF)


def test_validate_predictions_catches_null_field():
    records = [_record("a", counterparty=None), _record("b"), _record("c")]
    with pytest.raises(AssertionError, match="null"):
        validate_predictions(records, TEST_DF)
