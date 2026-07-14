"""Rule-based `recurring_flag` detector.

Phase 1 EDA (EDA_FINDINGS.md sec 2) found ZERO `I-RECURRING_FLAG` labels in
train or val -- there is no supervised signal for this field at all, so a
token classifier is structurally incapable of learning it, no matter how
it's weighted. But the underlying concept (recurring/subscription language)
does appear in the raw token text at a stable ~2.3% rate across train,
val, and test. This module replaces the model's (always-empty)
recurring_flag prediction with a keyword/regex match over the token text.

Kept as a separate, tiny module (not folded into fields.py) because it
operates on raw token text, not on predicted tags -- it's a fully
independent signal path, not a post-processing step over the classifier's
output for this field.
"""
from __future__ import annotations

import re

# Same keyword families used in eda.py's grep-based base-rate estimate:
# recur*, preauth*, subscri*, auto(debit|pay|approv|renew). The auto(...)
# group is deliberately narrow (not bare "auto") so it doesn't fire on
# unrelated text like "AUTOMATIC" or "AUTOMOBILE".
RECURRING_KEYWORD_RE = re.compile(
    r"(recur|preauth|subscri|auto(debit|pay|approv|renew))",
    re.IGNORECASE,
)


def detect_recurring(tokens: list[str]) -> str:
    """Return the space-joined matching token(s), or "" if none match.

    Returning the actual matched token(s) rather than a constant flag
    string keeps this consistent with extract_fields' convention (every
    scored field is "the text that justifies the tag", empty if absent)
    and with presence_score in eval_harness.py, which only checks
    non-empty vs non-empty.
    """
    hits = [tok for tok in tokens if RECURRING_KEYWORD_RE.search(tok)]
    return " ".join(hits)
