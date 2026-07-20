"""Model training: time-ordered split, imbalance-aware LightGBM, calibration.

Design decisions (each is an interview talking point and a Part B
validation checkpoint):

1. **Time-ordered split** (70/15/15 by timestamp), never random: application
   fraud drifts, and a random split would leak future ring members into
   training. The out-of-time test set is also what makes the later PSI
   drift analysis meaningful.
2. **Imbalance handled via scale_pos_weight**, not resampling: keeps the
   probability scale interpretable and avoids synthetic-oversampling
   artefacts that validators (rightly) flag.
3. **Isotonic calibration** fitted on the validation fold: fraud scores feed
   case-prioritisation and risk-appetite thresholds, so they must behave
   like probabilities, not just rank well.
"""

from __future__ import annotations

import warnings

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression

from .features import CATEGORICAL, feature_columns

RANDOM_STATE = 42


def time_split(df: pd.DataFrame, train_frac: float = 0.7, valid_frac: float = 0.15):
    n = len(df)
    i1, i2 = int(n * train_frac), int(n * (train_frac + valid_frac))
    return df.iloc[:i1], df.iloc[i1:i2], df.iloc[i2:]


def train_model(train: pd.DataFrame, valid: pd.DataFrame, cols: list[str] | None = None):
    cols = cols if cols is not None else feature_columns(train)
    x_tr, x_va = train[cols].copy(), valid[cols].copy()
    for c in CATEGORICAL:
        x_tr[c] = x_tr[c].astype("category")
        x_va[c] = x_va[c].astype("category")
    y_tr, y_va = train["is_fraud"].to_numpy(), valid["is_fraud"].to_numpy()

    pos_weight = (y_tr == 0).sum() / max(1, (y_tr == 1).sum())
    model = lgb.LGBMClassifier(
        objective="binary",
        n_estimators=2000,
        learning_rate=0.03,
        num_leaves=63,
        min_child_samples=50,
        subsample=0.8,
        subsample_freq=1,
        colsample_bytree=0.8,
        scale_pos_weight=pos_weight,
        random_state=RANDOM_STATE,
        verbose=-1,
    )
    with warnings.catch_warnings():
        # eval_set is deprecated in favour of eval_X/eval_y, but the new API
        # rejects DataFrame eval folds in lightgbm 4.x; keep the stable form.
        warnings.simplefilter("ignore")
        model.fit(
            x_tr, y_tr,
            eval_set=[(x_va, y_va)],
            eval_metric="average_precision",
            callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(0)],
        )

    calibrator = IsotonicRegression(out_of_bounds="clip")
    calibrator.fit(model.predict_proba(x_va)[:, 1], y_va)
    return model, calibrator, cols


def predict_calibrated(model, calibrator, df: pd.DataFrame, cols: list[str]) -> np.ndarray:
    x = df[cols].copy()
    for c in CATEGORICAL:
        x[c] = x[c].astype("category")
    return calibrator.predict(model.predict_proba(x)[:, 1])


def gain_importances(model, cols: list[str]) -> dict[str, float]:
    imp = model.booster_.feature_importance(importance_type="gain")
    return dict(zip(cols, imp.astype(float)))
