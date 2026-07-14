"""Central configuration: paths, URLs, tag set, and reproducibility.

Every script/notebook in this repo imports SEED from here and calls
set_seed() before doing anything stochastic (splits, shuffling, model
init, training). Keeping it in one place means "reproduce this run"
never depends on remembering to copy a magic number around.
"""
from __future__ import annotations

import os
import random
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
SEED = 42


def set_seed(seed: int = SEED) -> None:
    """Seed every RNG we know about. Safe to call even if torch isn't
    installed yet (Phase 0/1 don't need it)."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


# ---------------------------------------------------------------------------
# Data source
# ---------------------------------------------------------------------------
DATA_BASE_URL = "http://3.108.8.61:8990/datasets"
DATA_URLS = {
    "train": f"{DATA_BASE_URL}/train.jsonl",
    "val": f"{DATA_BASE_URL}/val.jsonl",
    "test": f"{DATA_BASE_URL}/test.jsonl",
}

REPO_ROOT = Path(__file__).resolve().parent
DATA_CACHE_DIR = REPO_ROOT / "data_cache"
CHECKPOINT_DIR = REPO_ROOT / "checkpoints"

# ---------------------------------------------------------------------------
# Tag set (flat I- scheme, no B-)
# ---------------------------------------------------------------------------
# The 4 tags whose extracted spans are actually scored on the leaderboard.
SCORED_TAGS = [
    "I-COUNTERPARTY_NAME",
    "I-TRANSACTION_METHOD",
    "I-PROCESSOR",
    "I-RECURRING_FLAG",
]

# Tags that exist in the annotation scheme but aren't part of the scored
# fields. "O" is included here but see the EDA notes: annotators used O
# both for true negatives *and* uncertain positives, so it's not a clean
# negative class.
UNSCORED_TAGS = [
    "O",
    "I-BANK_SERVICE_EVENT",
    "I-FILLER_WORD",
    "I-SEPARATOR_PUNCTUATION",
]

ALL_TAGS = UNSCORED_TAGS[:1] + SCORED_TAGS + UNSCORED_TAGS[1:]  # O first, matches spec order
assert set(ALL_TAGS) == set(SCORED_TAGS) | set(UNSCORED_TAGS)
assert len(ALL_TAGS) == 8

# Maps from the scored tag name to the submission JSON field name.
TAG_TO_FIELD = {
    "I-COUNTERPARTY_NAME": "counterparty",
    "I-TRANSACTION_METHOD": "transaction_method",
    "I-PROCESSOR": "processor",
    "I-RECURRING_FLAG": "recurring_flag",
}
FIELD_TO_TAG = {v: k for k, v in TAG_TO_FIELD.items()}
SCORED_FIELDS = list(TAG_TO_FIELD.values())

TAG2ID = {tag: i for i, tag in enumerate(ALL_TAGS)}
ID2TAG = {i: tag for tag, i in TAG2ID.items()}

# Expected corpus shape, used as an ingestion sanity check (Phase 0) and
# again after reconciliation (Phase 2). If the API's data ever changes
# shape, this assertion is meant to fail loudly rather than let a silent
# schema drift propagate into training.
EXPECTED_SHAPES = {
    "train": {"rows": 20000, "unique_ids": 10000, "annotators_per_id": 2},
    "val": {"rows": 2000, "unique_ids": 1000, "annotators_per_id": 2},
    "test": {"rows": 10000, "unique_ids": 10000, "annotators_per_id": 1},
}

# ---------------------------------------------------------------------------
# Annotator pool structure (found in Phase 1 EDA, see EDA_FINDINGS.md)
# ---------------------------------------------------------------------------
# Of the 5 annotator_ids, these 3 agree with EACH OTHER with zero token-level
# mismatches across every single train/val id where two of them co-annotate
# (5,980 ids / ~48k tokens in train alone -- verified exhaustively, not
# sampled). That's not plausible for independent human judgment calls
# (TRANSACTION_METHOD vs PROCESSOR has no strict rule per the annotator
# guidelines), so we treat these three as one deterministic labeling
# process rather than three independent raters.
#
# The other 2 annotators (ann_3adc6b, ann_8ac1bf) show real, human-like
# disagreement (kappa 0.60-0.82) both with the big three and with each
# other. Every single id in train/val has at least one BIG_THREE_ANNOTATORS
# member (the pair (ann_3adc6b, ann_8ac1bf) never co-occurs), so a
# BIG_THREE row is always available as a reconciliation anchor.
BIG_THREE_ANNOTATORS = {"ann_78009b", "ann_928994", "ann_f2aee8"}
MINORITY_ANNOTATORS = {"ann_3adc6b", "ann_8ac1bf"}
