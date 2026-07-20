"""Pillar 2 check: validator-side label-quality scan (confident learning).

Fraud labels are never clean in reality (delayed confirmation, missed fraud,
investigator error). CleanLab's confident-learning approach flags examples
whose labels disagree with out-of-sample model consensus. On synthetic data
the labels are correct by construction, so this run doubles as a
**false-positive-rate check of the tool itself** — how much noise does the
scanner report when the true noise is ~zero?
"""

from __future__ import annotations

import warnings

import lightgbm as lgb
import numpy as np
import pandas as pd
from cleanlab.filter import find_label_issues
from sklearn.model_selection import cross_val_predict

from ..features import CATEGORICAL


def label_issue_scan(train: pd.DataFrame, cols: list[str], sample: int = 25_000, seed: int = 42) -> dict:
    df = train.sample(n=min(sample, len(train)), random_state=seed)
    x = df[cols].copy()
    for c in CATEGORICAL:
        x[c] = x[c].astype("category").cat.codes  # sklearn CV needs numerics
    y = df["is_fraud"].to_numpy()

    clf = lgb.LGBMClassifier(
        objective="binary", n_estimators=300, learning_rate=0.05, num_leaves=31,
        scale_pos_weight=(y == 0).sum() / max(1, (y == 1).sum()),
        random_state=seed, verbose=-1,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        probs = cross_val_predict(clf, x, y, cv=3, method="predict_proba")
    issues = find_label_issues(y, probs, filter_by="prune_by_noise_rate")
    flagged = df.loc[issues]
    return {
        "n_scanned": int(len(df)),
        "n_flagged": int(issues.sum()),
        "flagged_rate": float(issues.mean()),
        "flagged_fraud_share": float(flagged["is_fraud"].mean()) if issues.sum() else 0.0,
        "flagged_by_type": flagged["fraud_type"].value_counts().to_dict() if issues.sum() else {},
    }
