"""Convert a (tokens, tags) sequence into the four scored submission
fields. Shared by eval_harness.py (scoring predictions during
development) and infer.py (Phase 5/7, building predictions.json) so
every path extracts fields identically.
"""
from __future__ import annotations

from config import SCORED_FIELDS, TAG_TO_FIELD


def extract_fields(tokens: list[str], tags: list[str]) -> dict[str, str]:
    """One space-joined string per scored field, empty string if absent.

    Design decision -- multiple non-contiguous spans of the same class
    (e.g. two separate processor mentions) are simply concatenated in
    original token order. This is deliberate: scoring is bag-of-words
    per transaction, so a single concatenated string and "properly"
    joined separate spans are indistinguishable under a multiset
    comparison -- there's no bag-of-words-visible difference, so plain
    concatenation loses nothing and avoids inventing an arbitrary
    span-separator convention the scorer was never told about.
    """
    buckets: dict[str, list[str]] = {f: [] for f in SCORED_FIELDS}
    for tok, tag in zip(tokens, tags):
        field = TAG_TO_FIELD.get(tag)
        if field is not None:
            buckets[field].append(tok)
    return {f: " ".join(toks) for f, toks in buckets.items()}
