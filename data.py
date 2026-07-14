"""Data ingestion for the Transaction NER assignment.

Downloads train/val/test JSONL from the dataset API, caches the raw
bytes locally (the API reshuffles line order on every call, but the
*content* is stable, so caching avoids re-downloading ~tens of
thousands of lines on every notebook restart), and parses each split
into a pandas DataFrame.

Everything downstream must join/group on `id`, never on line position.
To make that hard to get wrong by accident, we set `id` as the
DataFrame index... but then every consumer would need
`reset_index()` or `.loc`, which is its own footgun. Instead we keep
`id` as a normal column and rely on `validate_schema()` (run at import
time in tests, and explicitly in Phase 0) to catch any code that
silently assumes row order == id order.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import pandas as pd
import requests

from config import DATA_CACHE_DIR, DATA_URLS, EXPECTED_SHAPES

logger = logging.getLogger(__name__)

REQUEST_TIMEOUT_S = 60


def _download_raw(split: str, cache_dir: Path = DATA_CACHE_DIR, force: bool = False) -> Path:
    """Fetch the raw .jsonl bytes for `split` and cache to disk.

    Returns the local path. Does NOT re-download if a cached copy
    already exists, unless force=True -- the assignment brief warns
    line *order* is reshuffled per-download, but row content per id is
    fixed, so re-downloading buys nothing for train/val and only
    matters for test if the server-side test set itself changes.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    dest = cache_dir / f"{split}.jsonl"

    if dest.exists() and not force:
        logger.info("Using cached %s (%s)", split, dest)
        return dest

    url = DATA_URLS[split]
    logger.info("Downloading %s from %s", split, url)
    resp = requests.get(url, timeout=REQUEST_TIMEOUT_S)
    resp.raise_for_status()
    dest.write_bytes(resp.content)
    return dest


def _parse_jsonl(path: Path) -> list[dict]:
    records = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise ValueError(f"{path}:{line_no}: malformed JSON line") from e
    return records


def _records_to_frame(records: list[dict], split: str) -> pd.DataFrame:
    """Flatten the nested `metadata` dict into top-level columns and
    keep `tokens`/`ner_tags` as list-valued columns (object dtype) --
    they're sequences, not scalars, so we don't want pandas coercing
    them into anything else.
    """
    rows = []
    for rec in records:
        meta = rec.get("metadata", {})
        row = {
            "id": rec["id"],
            "annotator_id": rec.get("annotator_id"),  # absent in test
            "tokens": rec["tokens"],
            "ner_tags": rec.get("ner_tags"),  # absent in test
            "bank": meta.get("bank"),
            "case_id": meta.get("case_id"),
            "row_index": meta.get("row_index"),
            "date": meta.get("date"),
            "amount": meta.get("amount"),
            "transaction_type": meta.get("transaction_type"),
            "original_description": meta.get("original_description"),
        }
        rows.append(row)
    df = pd.DataFrame(rows)
    df.attrs["split"] = split
    return df


def validate_schema(df: pd.DataFrame, split: str) -> None:
    """Assert the frame matches the documented shape for `split`.

    Checks, in order of how likely they are to catch a real bug:
      1. Row / unique-id counts match EXPECTED_SHAPES.
      2. Every id appears exactly `annotators_per_id` times.
      3. For splits with 2 annotations per id (train/val), both rows
         for a given id share the *identical* tokens list -- the brief
         states tokenization is fixed across annotators, so if this
         ever fails it means either the API changed or our parsing is
         wrong, not that reconciliation needs to handle differing
         tokens.
      4. (train/val only) every ner_tags value is in the documented
         8-tag vocabulary.
    """
    expected = EXPECTED_SHAPES[split]

    assert len(df) == expected["rows"], (
        f"{split}: expected {expected['rows']} rows, got {len(df)}"
    )

    id_counts = df["id"].value_counts()
    assert id_counts.nunique() == 1 and id_counts.iloc[0] == expected["annotators_per_id"], (
        f"{split}: expected every id to appear exactly "
        f"{expected['annotators_per_id']} times, got counts {id_counts.value_counts().to_dict()}"
    )
    assert df["id"].nunique() == expected["unique_ids"], (
        f"{split}: expected {expected['unique_ids']} unique ids, got {df['id'].nunique()}"
    )

    if expected["annotators_per_id"] == 2:
        mismatches = []
        for tx_id, group in df.groupby("id"):
            tok_lists = group["tokens"].tolist()
            if tok_lists[0] != tok_lists[1]:
                mismatches.append(tx_id)
        assert not mismatches, (
            f"{split}: {len(mismatches)} ids have differing token lists across "
            f"annotators, e.g. {mismatches[:5]} -- tokenization was expected to be fixed"
        )

    if "ner_tags" in df.columns and df["ner_tags"].notna().any():
        from config import ALL_TAGS

        seen_tags = set()
        for tags in df["ner_tags"].dropna():
            seen_tags.update(tags)
        unexpected = seen_tags - set(ALL_TAGS)
        assert not unexpected, f"{split}: unexpected tags not in ALL_TAGS: {unexpected}"

    logger.info("%s: schema OK (%d rows, %d unique ids)", split, len(df), df["id"].nunique())


def load_split(split: str, cache_dir: Path = DATA_CACHE_DIR, force_download: bool = False,
                validate: bool = True) -> pd.DataFrame:
    """Download (or reuse cache), parse, and return one split as a DataFrame."""
    if split not in DATA_URLS:
        raise ValueError(f"Unknown split {split!r}, expected one of {list(DATA_URLS)}")
    path = _download_raw(split, cache_dir=cache_dir, force=force_download)
    records = _parse_jsonl(path)
    df = _records_to_frame(records, split)
    if validate:
        validate_schema(df, split)
    return df


def load_all(cache_dir: Path = DATA_CACHE_DIR, force_download: bool = False,
             validate: bool = True) -> dict[str, pd.DataFrame]:
    """Convenience wrapper: load train, val, and test in one call."""
    return {
        split: load_split(split, cache_dir=cache_dir, force_download=force_download, validate=validate)
        for split in ("train", "val", "test")
    }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    frames = load_all()
    for name, df in frames.items():
        print(f"\n=== {name} ===")
        print(df.shape)
        print(df.head(3))
