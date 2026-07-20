"""Evaluation for the fraud model — metrics chosen for the fraud-ops reality.

ROC-AUC is reported but never headlined: at ~2.5% prevalence it flatters.
The operational metrics are:
- **PR-AUC (average precision)** — headline discrimination metric.
- **Precision@k%** — if investigators can only review the top k% of
  applications, how many are actually fraud (case-prioritisation quality).
- **Recall@1% FPR** — how much fraud is caught at a tolerable false-alarm
  budget.
- **Brier score + calibration curve** — scores are only usable for
  risk-appetite decisions if they behave like probabilities.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)


def precision_at_k(y_true: np.ndarray, scores: np.ndarray, frac: float) -> float:
    k = max(1, int(len(scores) * frac))
    top = np.argsort(scores)[::-1][:k]
    return float(y_true[top].mean())


def recall_at_fpr(y_true: np.ndarray, scores: np.ndarray, max_fpr: float) -> float:
    fpr, tpr, _ = roc_curve(y_true, scores)
    return float(np.interp(max_fpr, fpr, tpr))


def compute_metrics(y_true: np.ndarray, scores: np.ndarray) -> dict:
    return {
        "pr_auc": float(average_precision_score(y_true, scores)),
        "roc_auc": float(roc_auc_score(y_true, scores)),
        "precision_at_0.5pct": precision_at_k(y_true, scores, 0.005),
        "precision_at_1pct": precision_at_k(y_true, scores, 0.01),
        "precision_at_5pct": precision_at_k(y_true, scores, 0.05),
        "recall_at_1pct_fpr": recall_at_fpr(y_true, scores, 0.01),
        "brier_score": float(brier_score_loss(y_true, scores)),
        "prevalence": float(y_true.mean()),
        "n": int(len(y_true)),
    }


def save_report(
    metrics: dict,
    y_true: np.ndarray,
    scores: np.ndarray,
    importances: dict[str, float],
    out_dir: str | Path,
) -> None:
    out = Path(out_dir)
    fig_dir = out / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2))

    prec, rec, _ = precision_recall_curve(y_true, scores)
    plt.figure(figsize=(6, 4.5))
    plt.plot(rec, prec)
    plt.axhline(metrics["prevalence"], ls="--", c="grey", label=f"prevalence {metrics['prevalence']:.3f}")
    plt.xlabel("Recall"); plt.ylabel("Precision")
    plt.title(f"Precision-Recall (AP = {metrics['pr_auc']:.3f})")
    plt.legend(); plt.tight_layout()
    plt.savefig(fig_dir / "pr_curve.png", dpi=150); plt.close()

    frac_pos, mean_pred = calibration_curve(y_true, scores, n_bins=10, strategy="quantile")
    plt.figure(figsize=(6, 4.5))
    plt.plot(mean_pred, frac_pos, marker="o", label="model")
    plt.plot([0, max(mean_pred)], [0, max(mean_pred)], ls="--", c="grey", label="perfect")
    plt.xlabel("Mean predicted probability"); plt.ylabel("Observed fraud rate")
    plt.title(f"Calibration (Brier = {metrics['brier_score']:.4f})")
    plt.legend(); plt.tight_layout()
    plt.savefig(fig_dir / "calibration_curve.png", dpi=150); plt.close()

    top = sorted(importances.items(), key=lambda kv: kv[1], reverse=True)[:15]
    names, vals = zip(*top)
    plt.figure(figsize=(7, 5))
    plt.barh(range(len(names)), vals)
    plt.yticks(range(len(names)), names)
    plt.gca().invert_yaxis()
    plt.title("Feature importance (gain, top 15)")
    plt.tight_layout()
    plt.savefig(fig_dir / "feature_importance.png", dpi=150); plt.close()
