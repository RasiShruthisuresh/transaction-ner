"""Label reconciliation: collapse each id's two annotation rows into one.

Primary strategy is decided by the Phase 1 EDA finding (EDA_FINDINGS.md
sec 4): every train/val id has >=1 annotator from BIG_THREE_ANNOTATORS,
and all big-three pairs agree on every token with zero mismatches
(verified exhaustively, not sampled). So reconciliation reduces to
"take the big-three row as gold" -- no per-token tie-break logic is
needed for the common case. Two alternative strategies from the brief
are implemented for comparison but are NOT the primary path.
"""
from __future__ import annotations

import pandas as pd

from config import BIG_THREE_ANNOTATORS


def reconcile_big_three(df: pd.DataFrame) -> pd.DataFrame:
    """Primary strategy. One row per id, taken from a BIG_THREE_ANNOTATORS
    member. Asserts the EDA finding holds (every id has >=1 big-three
    annotator) rather than silently falling back, since the whole point
    is that this precondition was verified, not assumed.
    """
    is_big_three = df["annotator_id"].isin(BIG_THREE_ANNOTATORS)
    ids_with_gold = set(df.loc[is_big_three, "id"])
    missing = set(df["id"]) - ids_with_gold
    assert not missing, f"{len(missing)} ids have no big-three annotator: {list(missing)[:5]}"

    gold = (
        df[is_big_three]
        .sort_values(["id", "annotator_id"])
        .drop_duplicates("id", keep="first")  # big-three rows for the same id are identical; any is fine
        .reset_index(drop=True)
    )
    return gold


def reconcile_duplication(df: pd.DataFrame) -> pd.DataFrame:
    """Alternative 1 (duplication-as-augmentation): keep both annotated
    rows per id as separate training examples. This is just `df` --
    named as a function so the alternative is visible/callable alongside
    the primary strategy rather than being an implicit "do nothing".
    """
    return df.reset_index(drop=True)


def reconcile_token_tiebreak(df: pd.DataFrame) -> pd.DataFrame:
    """Alternative 2 (per-token tie-break, ignoring the big-three finding):
    if both annotators agree on a token, use that tag. If they disagree
    and exactly one side is "O", prefer the non-O tag -- per annotator
    guideline #5, O is contaminated with uncertain positives, so a
    confident non-O guess from either annotator is treated as more
    informative than a same-token O. If both sides are non-O and
    disagree (rare -- the EDA confusion matrix shows scored tags almost
    never swap with each other), fall back to preferring the
    BIG_THREE_ANNOTATORS side.

    This differs from reconcile_big_three ONLY on tokens where the
    big-three annotator said O but the minority annotator gave a
    confident non-O guess: there, this strategy trusts the minority
    guess instead of keeping the big-three O. Kept for comparison, not
    used as the primary strategy -- trusting a single noisier human
    guess over a verified-consistent multi-annotator agreement is a
    real judgment call, and EDA didn't have positive evidence the
    minority annotators' non-O guesses under big-three-O were correct.
    """
    rows = []
    for tx_id, group in df.groupby("id"):
        if len(group) == 1:
            rows.append(group.iloc[0].to_dict())
            continue
        group = group.sort_values("annotator_id")
        a, b = group.iloc[0], group.iloc[1]
        tags = []
        for ta, tb in zip(a["ner_tags"], b["ner_tags"]):
            if ta == tb:
                tags.append(ta)
            elif ta == "O" and tb != "O":
                tags.append(tb)
            elif tb == "O" and ta != "O":
                tags.append(ta)
            else:
                tags.append(ta if a["annotator_id"] in BIG_THREE_ANNOTATORS else tb)
        out = a.to_dict()
        out["ner_tags"] = tags
        out["annotator_id"] = "reconciled_tiebreak"
        rows.append(out)
    return pd.DataFrame(rows).reset_index(drop=True)


def validate_reconciled(df: pd.DataFrame, source: pd.DataFrame, expected_rows: int) -> None:
    """Well-formedness check shared by all three strategies: right row
    count, one row per id, and tokens match the source id's tokens
    exactly (reconciliation must never invent or drop tokens)."""
    assert len(df) == expected_rows, f"expected {expected_rows} rows, got {len(df)}"
    assert df["id"].is_unique, "reconciled frame must have exactly one row per id"
    assert set(df["id"]) == set(source["id"]), "reconciled ids must match source ids exactly"

    source_tokens = source.drop_duplicates("id").set_index("id")["tokens"]
    for tx_id, tokens in zip(df["id"], df["tokens"]):
        assert tokens == source_tokens[tx_id], f"token mismatch for id={tx_id}"
        # ner_tags length must match tokens length (no dropped/added labels)
    for tokens, tags in zip(df["tokens"], df["ner_tags"]):
        assert len(tokens) == len(tags)


if __name__ == "__main__":
    from data import load_split

    for split, expected in (("train", 10000), ("val", 1000)):
        raw = load_split(split)
        gold = reconcile_big_three(raw)
        validate_reconciled(gold, raw, expected)
        print(f"{split}: reconcile_big_three OK -- {len(gold)} rows, all from BIG_THREE_ANNOTATORS")
