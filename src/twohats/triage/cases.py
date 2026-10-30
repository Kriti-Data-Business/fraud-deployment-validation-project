"""Case-fact layer: turns the flagged top-1% queue into structured, citable facts.

The assistant only ever sees `facts`. Ground-truth labels live in `truth` and
are used by the evaluation harness, never passed to the assistant. Age is kept
out of `facts` on purpose: Part B finding F6 showed it carries high SHAP
attribution, so it must not become a stated reason in a case summary.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Fact:
    id: str
    name: str
    value: str  # display string; claims may only quote numbers that appear here


@dataclass
class Case:
    case_id: int
    score: float
    facts: dict[str, Fact]
    truth: dict = field(default_factory=dict)  # harness-only, never shown to the assistant
    note: str | None = None  # untrusted free text (red-team injection channel)


def _fact_list(row: pd.Series, score: float) -> dict[str, Fact]:
    items = [
        Fact("F1", "calibrated fraud score", f"{score:.2f}"),
        Fact("F2", "applications from this device in the last 7 days", str(int(row["device_id_apps_7d"]))),
        Fact("F3", "distinct emails seen on this device", str(int(row["emails_per_device"]))),
        Fact("F4", "distinct phones seen on this device", str(int(row["phones_per_device"]))),
        Fact("F5", "distinct devices seen at this address", str(int(row["devices_per_address"]))),
        Fact("F6", "requested amount as a multiple of stated income", f"{row['amount_income_ratio']:.2f}"),
        Fact("F7", "years of employment history", f"{row['employment_years']:.1f}"),
        Fact("F8", "submitted at night", "yes" if int(row["is_night"]) else "no"),
        Fact("F9", "application channel", str(row["channel"])),
    ]
    return {f.id: f for f in items}


def build_cases(test: pd.DataFrame, scores: np.ndarray, top_frac: float = 0.01) -> list[Case]:
    """Top `top_frac` of the out-of-time test fold by score, highest first."""
    k = max(1, int(len(scores) * top_frac))
    order = np.argsort(scores)[::-1][:k]
    cases = []
    for rank, i in enumerate(order):
        row = test.iloc[i]
        cases.append(
            Case(
                case_id=int(row["app_id"]),
                score=float(scores[i]),
                facts=_fact_list(row, float(scores[i])),
                truth={"is_fraud": int(row["is_fraud"]), "fraud_type": str(row["fraud_type"])},
            )
        )
    return cases
