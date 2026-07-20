"""Pillar 2 check: explainability review (SHAP).

Global: does the model's attribution structure match its stated theory
(entity linkage + affordability)? Local: for top-queue cases, would the
explanation survive an investigator's or a customer's challenge?
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

from ..features import CATEGORICAL


def shap_review(model, test: pd.DataFrame, cols: list[str], scores: np.ndarray, fig_dir: str | Path) -> dict:
    fig_dir = Path(fig_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)

    x = test[cols].copy()
    for c in CATEGORICAL:
        x[c] = x[c].astype("category")
    sample_idx = np.random.default_rng(0).choice(len(x), size=min(3000, len(x)), replace=False)
    xs = x.iloc[sample_idx]

    explainer = shap.TreeExplainer(model.booster_)
    sv = explainer.shap_values(xs)
    if isinstance(sv, list):  # binary case returns [neg, pos] in some versions
        sv = sv[1]

    plt.figure()
    shap.summary_plot(sv, xs, show=False, max_display=14, plot_size=(8, 6))
    plt.tight_layout()
    plt.savefig(fig_dir / "shap_summary.png", dpi=150, bbox_inches="tight")
    plt.close("all")

    mean_abs = pd.Series(np.abs(sv).mean(axis=0), index=cols).sort_values(ascending=False)

    # Local explanations for the 3 highest-scored applications
    top3 = np.argsort(scores)[::-1][:3]
    locals_ = []
    sv_top = explainer.shap_values(x.iloc[top3])
    if isinstance(sv_top, list):
        sv_top = sv_top[1]
    for row, contribs in zip(top3, sv_top):
        s = pd.Series(contribs, index=cols).sort_values(key=np.abs, ascending=False)[:5]
        locals_.append({
            "app_index": int(row),
            "score": float(scores[row]),
            "actual_fraud_type": str(test.iloc[row]["fraud_type"]),
            "top_drivers": {k: round(float(v), 3) for k, v in s.items()},
        })

    return {
        "global_top10_mean_abs_shap": {k: round(float(v), 4) for k, v in mean_abs[:10].items()},
        "top_queue_local_explanations": locals_,
    }
