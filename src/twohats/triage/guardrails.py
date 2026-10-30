"""Deterministic guardrails. Each check returns a violation code or nothing.

The layer fails closed: any violation replaces the assistant output with a
fallback that routes the case to a human with no AI summary. The guardrails are
themselves tested (tests/test_triage.py and the seeded-fault run), because an
untested control is only a belief.

Known limit, measured in docs/part_c_validation.md: these checks verify form,
citations, numbers, language and consistency. They cannot tell whether a
sentence without numbers is true. That needs a semantic judge, which is the
next layer to add.
"""

from __future__ import annotations

import re

from .assistant import ALLOWED_NEXT_STEPS, HIGH_SCORE, PATTERNS, RISK_LEVELS, derive_pattern
from .cases import Case

NUMBER = re.compile(r"\d+(?:\.\d+)?")
FACT_REF = re.compile(r"\bF\d+\b")  # a citation such as "F3" is a reference, not a numeric claim
DECISION = re.compile(
    r"\b(approve[sd]?|approval|decline[sd]?|declining|reject(?:ed|ing|s)?|deny|denied|denying|"
    r"refuse[sd]?|should be (?:approved|declined|rejected)|grant(?:ed)?)\b",
    re.I,
)
PROTECTED = re.compile(r"\b(age|aged|young|younger|old|older|elderly|gender|male|female|ethnic\w*|race|religio\w*)\b", re.I)
PII = re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+|\b(?:rd|ra|rp|d|a|p)\d+(?:_\d+)?\b", re.I)
REQUIRED = {"risk_level", "pattern", "claims", "next_step", "human_review_required"}

FALLBACK = {
    "risk_level": None,
    "pattern": None,
    "claims": [],
    "next_step": "manual_review",
    "human_review_required": True,
    "fallback": True,
}


def _schema(out: dict) -> list[str]:
    if not isinstance(out, dict) or not REQUIRED <= set(out):
        return ["schema"]
    if out["risk_level"] not in RISK_LEVELS or out["pattern"] not in PATTERNS:
        return ["schema"]
    claims = out["claims"]
    if not isinstance(claims, list) or not claims:
        return ["schema"]
    for c in claims:
        if not isinstance(c, dict) or not isinstance(c.get("text"), str) or not isinstance(c.get("fact_ids"), list):
            return ["schema"]
    return []


def check(case: Case, out: dict, rederive: bool = True) -> list[str]:
    """Return the sorted list of violated guardrail codes (empty means pass).

    `rederive=False` switches off the pattern re-derivation check so the harness
    can measure what the form-level checks alone would have caught.
    """
    bad = _schema(out)
    if bad:
        return bad  # nothing else is checkable on a malformed output
    v: set[str] = set()
    text = " ".join(c["text"] for c in out["claims"])

    for c in out["claims"]:
        ids = c["fact_ids"]
        if not ids or any(i not in case.facts for i in ids):
            v.add("citation")
            continue
        allowed = {n for i in ids for n in NUMBER.findall(case.facts[i].value)}
        if any(n not in allowed for n in NUMBER.findall(FACT_REF.sub(" ", c["text"]))):
            v.add("ungrounded_number")

    if DECISION.search(text):
        v.add("decision_language")
    if PROTECTED.search(text):
        v.add("protected_attribute")
    if PII.search(text):
        v.add("pii_or_identifier")
    if out["human_review_required"] is not True:
        v.add("human_review")
    if out["next_step"] not in ALLOWED_NEXT_STEPS:
        v.add("next_step")
    if (case.score >= HIGH_SCORE) != (out["risk_level"] == "high"):
        v.add("score_contradiction")
    if case.note and _echoes(case.note, text):
        v.add("injection_echo")
    # Asymmetric on purpose: a summary may not lower scrutiny the facts support.
    # Escalating beyond the rule is allowed (a human reviews it either way).
    if rederive and out["pattern"] == "unclear" and derive_pattern(case) != "unclear":
        v.add("pattern_downgrade")
    return sorted(v)


def _echoes(note: str, text: str) -> bool:
    """True if any 4-word run of the untrusted note is repeated in the output."""
    words = re.findall(r"\w+", note.lower())
    out = " ".join(re.findall(r"\w+", text.lower()))
    return any(" ".join(words[i : i + 4]) in out for i in range(max(0, len(words) - 3)))


def guarded(assistant, case: Case) -> tuple[dict, list[str]]:
    """Run the assistant, apply guardrails, fail closed to the fallback."""
    out = assistant.generate(case)
    violations = check(case, out)
    return (FALLBACK if violations else out), violations
