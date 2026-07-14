"""Exploratory data analysis for the Transaction NER assignment.

Every function here returns a plain pandas object (Series/DataFrame) so
it's independently testable (see tests/test_eda.py) and reusable from a
notebook. `main()` just orchestrates: run everything, dump tables/plots
to eda_outputs/, and let EDA_FINDINGS.md narrate what they mean.

Design note on which split we analyze: EDA and agreement stats below
run on **train** (10k transactions, the bulk of the data available for
reconciliation decisions). val is loaded too and spot-checked for
consistency, but isn't the primary subject -- it exists to validate
generalization, not to characterize the label distribution.
"""
from __future__ import annotations

from collections import Counter

import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score, confusion_matrix

from config import ALL_TAGS, BIG_THREE_ANNOTATORS, SCORED_TAGS, TAG_TO_FIELD, UNSCORED_TAGS
from data import load_split

# ---------------------------------------------------------------------------
# 1. Token-level flattening
# ---------------------------------------------------------------------------


def explode_tokens(df: pd.DataFrame) -> pd.DataFrame:
    """One row per (transaction, annotator, token position).

    Keeps `id`, `annotator_id`, `bank`, `transaction_type` alongside
    each token/tag so downstream groupbys don't need to re-join.
    """
    records = []
    for row in df.itertuples(index=False):
        for pos, (tok, tag) in enumerate(zip(row.tokens, row.ner_tags)):
            records.append(
                {
                    "id": row.id,
                    "annotator_id": row.annotator_id,
                    "pos": pos,
                    "token": tok,
                    "token_norm": tok.strip().lower(),
                    "tag": tag,
                    "bank": row.bank,
                    "transaction_type": row.transaction_type,
                }
            )
    return pd.DataFrame.from_records(records)


# ---------------------------------------------------------------------------
# 2. Label distribution: token-level frequency, scored vs unscored
# ---------------------------------------------------------------------------


def tag_frequency(exploded: pd.DataFrame) -> pd.DataFrame:
    """Token count and share per tag, flagged scored vs unscored.

    This is the raw ingredient for the "training distribution vs what's
    scored" comparison the brief asks for: SCORED_TAGS make up a
    minority of tokens (most of a description is separators, filler,
    and O), and within SCORED_TAGS the four fields are far from evenly
    represented.
    """
    counts = exploded["tag"].value_counts()
    out = counts.reindex(ALL_TAGS).fillna(0).astype(int).rename("token_count").to_frame()
    out["token_share"] = out["token_count"] / out["token_count"].sum()
    out["scored"] = out.index.isin(SCORED_TAGS)
    return out.sort_values("token_count", ascending=False)


def top_tokens_per_tag(exploded: pd.DataFrame, n: int = 15) -> dict[str, pd.Series]:
    """Most frequent normalized tokens for each tag -- sanity-check that
    e.g. I-RECURRING_FLAG really is a near-closed vocabulary."""
    out = {}
    for tag in ALL_TAGS:
        vc = exploded.loc[exploded["tag"] == tag, "token_norm"].value_counts().head(n)
        out[tag] = vc
    return out


# ---------------------------------------------------------------------------
# 3. Token-level frequency vs transaction-level presence (the key EDA hint)
# ---------------------------------------------------------------------------


def field_presence_rates(df: pd.DataFrame, exploded: pd.DataFrame) -> pd.DataFrame:
    """For each scored field, compare:
      - token_share: fraction of ALL tokens carrying this tag
      - row_presence_rate: fraction of annotation ROWS (one per
        transaction x annotator) that contain >=1 token of this tag
      - txn_presence_rate: fraction of unique TRANSACTIONS (ids) where
        AT LEAST ONE of the two annotators marked this tag anywhere

    Why this matters: recurring_flag scoring is presence-based (does
    the prediction have *any* non-empty span, matched against whether
    gold has one) -- so what drives that F1 is txn_presence_rate, not
    token_share. A tag can be token-rare but transaction-common (e.g.
    "recurring" appears as 1 token out of ~8, but if most
    recurring-flagged transactions only ever contain one such token,
    token_share and txn_presence_rate end up close). Conversely if a
    tag's tokens cluster into a few transactions with long spans,
    token_share overstates how many transactions actually have it.
    We compute both so we can see which pattern holds for each field.
    """
    total_tokens = len(exploded)
    total_rows = len(df)
    total_ids = df["id"].nunique()

    rows_out = []
    for tag, field in TAG_TO_FIELD.items():
        tok_count = (exploded["tag"] == tag).sum()
        rows_with_tag = exploded.loc[exploded["tag"] == tag, ["id", "annotator_id"]].drop_duplicates()
        row_presence = len(rows_with_tag) / total_rows
        ids_with_tag = exploded.loc[exploded["tag"] == tag, "id"].nunique()
        txn_presence = ids_with_tag / total_ids
        rows_out.append(
            {
                "field": field,
                "tag": tag,
                "token_share": tok_count / total_tokens,
                "row_presence_rate": row_presence,
                "txn_presence_rate": txn_presence,
                # ratio > 1 means token-rare-but-transaction-common (good --
                # model gets a usable signal per transaction even off few
                # tokens); ratio < 1 means tokens cluster into fewer
                # transactions with longer spans than presence alone suggests.
                "presence_to_token_ratio": txn_presence / (tok_count / total_tokens) if tok_count else np.nan,
            }
        )
    return pd.DataFrame(rows_out).set_index("field")


# ---------------------------------------------------------------------------
# 4. Inter-annotator agreement
# ---------------------------------------------------------------------------


def _id_pair_tag_sequences(df: pd.DataFrame) -> list[tuple[str, str, str, list[str], list[str]]]:
    """For every id with exactly 2 annotation rows, return
    (id, annotator_a, annotator_b, tags_a, tags_b) with annotator_a <
    annotator_b (stable ordering) and tags aligned by token position.
    Relies on data.validate_schema already having confirmed both rows
    share identical `tokens`.
    """
    out = []
    for tx_id, group in df.groupby("id"):
        if len(group) != 2:
            continue
        group = group.sort_values("annotator_id")
        a, b = group.iloc[0], group.iloc[1]
        out.append((tx_id, a.annotator_id, b.annotator_id, list(a.ner_tags), list(b.ner_tags)))
    return out


def annotator_agreement(df: pd.DataFrame) -> pd.DataFrame:
    """Per-id token-level simple agreement rate and pooled-pair Cohen's
    kappa, one row per id.

    Simple agreement (fraction of matching token tags) is reported per
    id because it's well-defined even for short sequences. Cohen's
    kappa is NOT computed per id here -- kappa on an 8-token sequence
    with little tag variety is noisy/undefined (e.g. a sequence that's
    entirely "O" for both annotators has undefined kappa, since
    there's no variance to correct chance agreement for). Kappa is
    instead pooled per annotator-pair in `pairwise_agreement_summary`.
    """
    rows = []
    for tx_id, a_id, b_id, tags_a, tags_b in _id_pair_tag_sequences(df):
        matches = sum(t1 == t2 for t1, t2 in zip(tags_a, tags_b))
        rows.append(
            {
                "id": tx_id,
                "annotator_a": a_id,
                "annotator_b": b_id,
                "n_tokens": len(tags_a),
                "n_matches": matches,
                "agreement_rate": matches / len(tags_a),
            }
        )
    return pd.DataFrame(rows)


def pairwise_agreement_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Pool all tokens per annotator-pair and compute one Cohen's kappa
    + simple agreement rate per pair (statistically sound, unlike
    averaging per-id kappas -- see annotator_agreement docstring)."""
    pooled: dict[tuple[str, str], tuple[list[str], list[str]]] = {}
    id_counts: Counter[tuple[str, str]] = Counter()
    for tx_id, a_id, b_id, tags_a, tags_b in _id_pair_tag_sequences(df):
        key = (a_id, b_id)
        pooled.setdefault(key, ([], []))
        pooled[key][0].extend(tags_a)
        pooled[key][1].extend(tags_b)
        id_counts[key] += 1

    rows = []
    for (a_id, b_id), (ta, tb) in pooled.items():
        kappa = cohen_kappa_score(ta, tb, labels=ALL_TAGS)
        agree = np.mean([x == y for x, y in zip(ta, tb)])
        rows.append(
            {
                "annotator_a": a_id,
                "annotator_b": b_id,
                "n_ids": id_counts[(a_id, b_id)],
                "n_tokens": len(ta),
                "agreement_rate": agree,
                "cohen_kappa": kappa,
            }
        )
    return pd.DataFrame(rows).sort_values("cohen_kappa")


def per_annotator_quality(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate agreement by single annotator_id (averaged across all
    partners they were paired with) to spot systematic outliers, plus
    each annotator's own O-rate as a "conservativeness" signal (recall
    the annotator guideline: uncertain -> O, so a much higher O-rate
    isn't necessarily wrong, but it does mean that annotator's non-O
    labels should be trusted more and their O's trusted less).
    """
    pair_summary = pairwise_agreement_summary(df)
    per_id_agreement = annotator_agreement(df)

    long = pd.concat(
        [
            per_id_agreement[["annotator_a", "agreement_rate"]].rename(columns={"annotator_a": "annotator_id"}),
            per_id_agreement[["annotator_b", "agreement_rate"]].rename(columns={"annotator_b": "annotator_id"}),
        ]
    )
    avg_agreement = long.groupby("annotator_id")["agreement_rate"].mean().rename("mean_agreement_with_partners")

    exploded = explode_tokens(df)
    tag_shares = (
        exploded.groupby(["annotator_id", "tag"]).size().unstack(fill_value=0)
    )
    tag_shares = tag_shares.div(tag_shares.sum(axis=1), axis=0)
    o_rate = tag_shares["O"].rename("o_rate") if "O" in tag_shares else pd.Series(dtype=float, name="o_rate")

    out = pd.concat([avg_agreement, o_rate], axis=1)
    out["n_rows_annotated"] = df["annotator_id"].value_counts()
    return out.sort_values("mean_agreement_with_partners")


def pooled_confusion_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Symmetric confusion matrix pooled across all annotator pairs and
    both orderings within a pair, to see *what* gets confused (e.g.
    PROCESSOR vs TRANSACTION_METHOD, or O vs COUNTERPARTY_NAME) rather
    than just how often."""
    all_a, all_b = [], []
    for _, _, _, tags_a, tags_b in _id_pair_tag_sequences(df):
        all_a.extend(tags_a + tags_b)
        all_b.extend(tags_b + tags_a)  # symmetrize
    cm = confusion_matrix(all_a, all_b, labels=ALL_TAGS)
    return pd.DataFrame(cm, index=ALL_TAGS, columns=ALL_TAGS)


def big_three_finding(df: pd.DataFrame) -> dict:
    """Exhaustively verify the BIG_THREE_ANNOTATORS finding from config.py:
    (1) every id has >=1 big-three annotator, and (2) every id annotated
    by two big-three members has zero token-level mismatches between
    them. Run over the full split (not a sample) since this claim is
    the basis for the Phase 2 reconciliation strategy and needs to be
    airtight, not "probably true."
    """
    id_annotators = df.groupby("id")["annotator_id"].apply(set)
    ids_without_big_three = id_annotators[~id_annotators.apply(lambda s: bool(s & BIG_THREE_ANNOTATORS))]

    mismatches = 0
    big_three_pairs_checked = 0
    for tx_id, a_id, b_id, tags_a, tags_b in _id_pair_tag_sequences(df):
        if {a_id, b_id} <= BIG_THREE_ANNOTATORS:
            big_three_pairs_checked += 1
            if tags_a != tags_b:
                mismatches += 1

    return {
        "n_ids": len(id_annotators),
        "ids_without_big_three_annotator": len(ids_without_big_three),
        "big_three_pairs_checked": big_three_pairs_checked,
        "big_three_pairs_with_any_mismatch": mismatches,
    }


# ---------------------------------------------------------------------------
# 5. Metadata clustering: bank / transaction_type vs scored-field vocab
# ---------------------------------------------------------------------------


def bank_field_concentration(exploded: pd.DataFrame, tag: str, min_bank_tokens: int = 10) -> pd.DataFrame:
    """For a given scored tag, and for each bank with >= min_bank_tokens
    tokens of that tag, report vocab size and "top-1 share" (fraction
    of that bank's tokens for this tag taken by the single most common
    token). High top-1 share + small vocab => bank is a strong feature
    for this field (e.g. every PROCESSOR mention at this bank is
    literally the same string) => worth feeding metadata.bank into the
    model. Low concentration => bank doesn't help much for this field.
    """
    sub = exploded[exploded["tag"] == tag]
    columns = ["bank", "n_tokens", "vocab_size", "top1_token", "top1_share"]
    rows = []
    for bank, group in sub.groupby("bank"):
        if len(group) < min_bank_tokens:
            continue
        vc = group["token_norm"].value_counts()
        rows.append(
            {
                "bank": bank,
                "n_tokens": len(group),
                "vocab_size": vc.shape[0],
                "top1_token": vc.index[0],
                "top1_share": vc.iloc[0] / len(group),
            }
        )
    if not rows:
        # No bank has >= min_bank_tokens tokens of this tag (expected for
        # rare tags like I-RECURRING_FLAG) -- return an empty-but-typed frame
        # rather than crashing on sort_values against a missing column.
        return pd.DataFrame(columns=columns)
    return pd.DataFrame(rows).sort_values("top1_share", ascending=False)


def transaction_type_field_presence(df: pd.DataFrame, exploded: pd.DataFrame) -> pd.DataFrame:
    """Presence rate of each scored field broken out by DEBIT vs CREDIT,
    to check whether transaction_type is informative (e.g. maybe
    recurring_flag is far more common on DEBIT than CREDIT)."""
    id_to_type = df.drop_duplicates("id").set_index("id")["transaction_type"]
    rows = []
    for tag, field in TAG_TO_FIELD.items():
        ids_with_tag = set(exploded.loc[exploded["tag"] == tag, "id"])
        types_with = id_to_type.loc[list(ids_with_tag)].value_counts()
        types_total = id_to_type.value_counts()
        for txn_type in types_total.index:
            rows.append(
                {
                    "field": field,
                    "transaction_type": txn_type,
                    "presence_rate": types_with.get(txn_type, 0) / types_total[txn_type],
                }
            )
    return pd.DataFrame(rows).pivot(index="field", columns="transaction_type", values="presence_rate")


# ---------------------------------------------------------------------------
# main: run everything, save artifacts
# ---------------------------------------------------------------------------


def main(out_dir: str = "eda_outputs") -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from pathlib import Path

    out = Path(out_dir)
    out.mkdir(exist_ok=True)

    train = load_split("train")
    val = load_split("val")
    exploded = explode_tokens(train)

    freq = tag_frequency(exploded)
    freq.to_csv(out / "tag_frequency.csv")

    presence = field_presence_rates(train, exploded)
    presence.to_csv(out / "field_presence_rates.csv")

    pair_summary = pairwise_agreement_summary(train)
    pair_summary.to_csv(out / "pairwise_agreement.csv", index=False)

    quality = per_annotator_quality(train)
    quality.to_csv(out / "per_annotator_quality.csv")

    cm = pooled_confusion_matrix(train)
    cm.to_csv(out / "confusion_matrix.csv")

    tx_type_presence = transaction_type_field_presence(train, exploded)
    tx_type_presence.to_csv(out / "transaction_type_field_presence.csv")

    for tag in SCORED_TAGS:
        conc = bank_field_concentration(exploded, tag)
        conc.to_csv(out / f"bank_concentration_{TAG_TO_FIELD[tag]}.csv", index=False)

    big_three_train = big_three_finding(train)
    big_three_val = big_three_finding(val)
    print("\n--- big_three_finding (train) ---")
    print(big_three_train)
    print("--- big_three_finding (val) ---")
    print(big_three_val)
    assert big_three_train["ids_without_big_three_annotator"] == 0
    assert big_three_train["big_three_pairs_with_any_mismatch"] == 0
    assert big_three_val["ids_without_big_three_annotator"] == 0
    assert big_three_val["big_three_pairs_with_any_mismatch"] == 0
    pd.DataFrame([{"split": "train", **big_three_train}, {"split": "val", **big_three_val}]).to_csv(
        out / "big_three_finding.csv", index=False
    )

    # -- plots --
    fig, ax = plt.subplots(figsize=(8, 5))
    freq["token_count"].plot(kind="barh", ax=ax, color=["#4C72B0" if s else "#999999" for s in freq["scored"]])
    ax.set_xlabel("token count")
    ax.set_title("Token count per tag (blue = scored fields)")
    fig.tight_layout()
    fig.savefig(out / "tag_frequency.png", dpi=120)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 5))
    presence[["token_share", "txn_presence_rate"]].plot(kind="bar", ax=ax)
    ax.set_ylabel("share / rate")
    ax.set_title("Token share vs transaction-level presence rate per scored field")
    fig.tight_layout()
    fig.savefig(out / "field_presence_vs_token_share.png", dpi=120)
    plt.close(fig)

    print(f"EDA artifacts written to {out}/")
    print("\n--- tag_frequency ---")
    print(freq)
    print("\n--- field_presence_rates ---")
    print(presence)
    print("\n--- pairwise_agreement_summary ---")
    print(pair_summary)
    print("\n--- per_annotator_quality ---")
    print(quality)


if __name__ == "__main__":
    main()
