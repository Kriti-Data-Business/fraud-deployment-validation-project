"""Pillar 3 checks: population/score stability (PSI) between time windows.

PSI conventions used across credit/fraud risk: < 0.1 stable, 0.1-0.25
investigate, > 0.25 material shift requiring action. These are industry
heuristics, not governance-calibrated thresholds — the report says so.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def psi(expected: np.ndarray, actual: np.ndarray, bins: int = 10) -> float:
    """Population Stability Index between two numeric samples."""
    qs = np.quantile(expected, np.linspace(0, 1, bins + 1))
    qs[0], qs[-1] = -np.inf, np.inf
    qs = np.unique(qs)
    if len(qs) < 3:  # degenerate (near-constant) feature
        return 0.0
    e_frac = np.histogram(expected, qs)[0] / max(1, len(expected))
    a_frac = np.histogram(actual, qs)[0] / max(1, len(actual))
    e_frac = np.clip(e_frac, 1e-6, None)
    a_frac = np.clip(a_frac, 1e-6, None)
    return float(np.sum((a_frac - e_frac) * np.log(a_frac / e_frac)))


def rate_psi(value: float) -> str:
    if value < 0.1:
        return "stable"
    if value < 0.25:
        return "investigate"
    return "action"


def feature_psi_table(
    train: pd.DataFrame, test: pd.DataFrame, cols: list[str]
) -> pd.DataFrame:
    rows = []
    for c in cols:
        if not pd.api.types.is_numeric_dtype(train[c]):
            continue
        v = psi(train[c].to_numpy(float), test[c].to_numpy(float))
        rows.append({"feature": c, "psi": round(v, 4), "rating": rate_psi(v)})
    return pd.DataFrame(rows).sort_values("psi", ascending=False, ignore_index=True)
