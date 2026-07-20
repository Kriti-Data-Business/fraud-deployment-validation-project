"""Pillar 2 checks: sensitivity, stress and feature-family ablation.

- **Input perturbation**: misstatement of income is the attack surface of an
  application form — how much does a ±20% income shift move scores and the
  top-1% queue?
- **Feature-family ablation**: retrain without a family; if performance
  collapses, that family is a single point of failure (an adversary's
  target and a data-outage risk).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..evaluate import compute_metrics
from ..features import ENTITY_COLS
from ..train import predict_calibrated, train_model

AFFORDABILITY = ["income", "amount_income_ratio", "employment_age_ratio", "employment_years"]
LINKAGE = (
    ["emails_per_device", "devices_per_address", "phones_per_device"]
    + [f"{c}_apps_{d}d" for c in ENTITY_COLS for d in (7, 30)]
)


def income_perturbation(model, calibrator, test: pd.DataFrame, cols: list[str]) -> dict:
    """Shift stated income ±20% and recompute the income-derived features.

    Entity-velocity features are unaffected by income, so recomputing the
    derived ratios on the featured frame is exact — no need to rebuild the
    point-in-time history.
    """
    base = predict_calibrated(model, calibrator, test, cols)
    out = {}
    for pct in (+0.2, -0.2):
        shifted = test.copy()
        shifted["income"] = shifted["income"] * (1 + pct)
        shifted["amount_income_ratio"] = shifted["requested_amount"] / (shifted["income"] + 1.0)
        s = predict_calibrated(model, calibrator, shifted, cols)
        k = max(1, int(len(s) * 0.01))
        base_top = set(np.argsort(base)[::-1][:k])
        new_top = set(np.argsort(s)[::-1][:k])
        out[f"income{pct:+.0%}"] = {
            "mean_abs_score_shift": float(np.mean(np.abs(s - base))),
            "top1pct_queue_turnover": float(1 - len(base_top & new_top) / k),
        }
    return out


def family_ablation(train: pd.DataFrame, valid: pd.DataFrame, test: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    y_te = test["is_fraud"].to_numpy()
    rows = []
    for name, drop in [
        ("full model", []),
        ("without entity-linkage family", LINKAGE),
        ("without affordability family", AFFORDABILITY),
    ]:
        keep = [c for c in cols if c not in drop]
        model, calibrator, _ = train_model(train, valid, cols=keep)
        scores = predict_calibrated(model, calibrator, test, keep)
        m = compute_metrics(y_te, scores)
        rows.append({
            "variant": name,
            "pr_auc": round(m["pr_auc"], 4),
            "precision_at_1pct": round(m["precision_at_1pct"], 4),
            "recall_at_1pct_fpr": round(m["recall_at_1pct_fpr"], 4),
        })
    return pd.DataFrame(rows)
