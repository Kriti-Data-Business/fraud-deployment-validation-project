"""Pillar 2 check: challenger benchmarking.

The champion must earn its complexity. Two independent challengers:

- **Regularised logistic regression** — the interpretable floor. If it gets
  close to the champion, complexity isn't buying enough.
- **Alternative GBM configuration** — shallow/fast variant, tests whether the
  champion's specific hyperparameters matter or the signal is easy.
"""

from __future__ import annotations

import warnings

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer

from ..evaluate import compute_metrics
from ..features import CATEGORICAL


def run_challengers(
    train: pd.DataFrame, test: pd.DataFrame, cols: list[str], champion_scores: np.ndarray
) -> pd.DataFrame:
    y_tr = train["is_fraud"].to_numpy()
    y_te = test["is_fraud"].to_numpy()
    num_cols = [c for c in cols if c not in CATEGORICAL]

    logreg = Pipeline(
        [
            ("prep", ColumnTransformer(
                [
                    ("num", StandardScaler(), num_cols),
                    ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
                ]
            )),
            ("clf", LogisticRegression(max_iter=2000, C=0.5, class_weight="balanced")),
        ]
    )
    logreg.fit(train[cols], y_tr)
    lr_scores = logreg.predict_proba(test[cols])[:, 1]

    alt = lgb.LGBMClassifier(
        objective="binary", n_estimators=300, learning_rate=0.1, num_leaves=15,
        min_child_samples=100, scale_pos_weight=(y_tr == 0).sum() / max(1, (y_tr == 1).sum()),
        random_state=7, verbose=-1,
    )
    x_tr, x_te = train[cols].copy(), test[cols].copy()
    for c in CATEGORICAL:
        x_tr[c] = x_tr[c].astype("category")
        x_te[c] = x_te[c].astype("category")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        alt.fit(x_tr, y_tr)
    alt_scores = alt.predict_proba(x_te)[:, 1]

    rows = []
    for name, scores in [
        ("champion (calibrated LightGBM)", champion_scores),
        ("challenger: logistic regression", lr_scores),
        ("challenger: shallow GBM", alt_scores),
    ]:
        m = compute_metrics(y_te, scores)
        rows.append({
            "model": name,
            "pr_auc": round(m["pr_auc"], 4),
            "precision_at_1pct": round(m["precision_at_1pct"], 4),
            "recall_at_1pct_fpr": round(m["recall_at_1pct_fpr"], 4),
        })
    return pd.DataFrame(rows)
