"""Generate README-ready charts from reports/validation/validation_results.json:
challenger comparison, feature-family ablation, segment capture rate.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RESULTS = json.loads(Path("reports/validation/validation_results.json").read_text())
OUT = Path("reports/validation/figures")
OUT.mkdir(parents=True, exist_ok=True)

NAVY, RED, GREEN, GREY = "#0B1F3A", "#E8555A", "#2FA36B", "#98A2B3"


def challenger_chart():
    rows = RESULTS["challengers"]
    labels = ["Champion\n(calibrated LightGBM)", "Challenger\n(logistic regression)", "Challenger\n(shallow GBM)"]
    values = [r["pr_auc"] for r in rows]
    colors = [NAVY, GREY, RED]  # red highlights the challenger that beat the champion
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    bars = ax.bar(labels, values, color=colors, width=0.55)
    ax.set_ylim(0.75, 0.92)
    ax.set_ylabel("PR-AUC (test fold)")
    ax.set_title("Challenger benchmark: a simpler model beat the champion")
    for b, v in zip(bars, values):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.004, f"{v:.3f}", ha="center", fontsize=11, fontweight="bold")
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    plt.savefig(OUT / "challenger_comparison.png", dpi=150)
    plt.close()


def ablation_chart():
    rows = RESULTS["family_ablation"]
    labels = ["Full model", "Without\nentity-linkage", "Without\naffordability"]
    values = [r["pr_auc"] for r in rows]
    colors = [NAVY, GREY, RED]
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    bars = ax.bar(labels, values, color=colors, width=0.55)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("PR-AUC (test fold)")
    ax.set_title("Feature-family ablation: one family carries the model")
    for b, v in zip(bars, values):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.02, f"{v:.3f}", ha="center", fontsize=11, fontweight="bold")
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    plt.savefig(OUT / "feature_ablation.png", dpi=150)
    plt.close()


def segment_chart():
    seg = RESULTS["segments"]
    labels = ["Ring fraud", "Opportunistic\nfraud"]
    values = [seg["ring"]["top1pct_capture"], seg["opportunistic"]["top1pct_capture"]]
    colors = [GREEN, RED]
    fig, ax = plt.subplots(figsize=(5.5, 4.2))
    bars = ax.bar(labels, values, color=colors, width=0.5)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Captured in top-1% queue")
    ax.set_title("Segment blind spot: solo fraud is under-detected")
    for b, v in zip(bars, values):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.02, f"{v:.1%}", ha="center", fontsize=12, fontweight="bold")
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    plt.savefig(OUT / "segment_capture.png", dpi=150)
    plt.close()


if __name__ == "__main__":
    challenger_chart()
    ablation_chart()
    segment_chart()
    print(f"Wrote 3 charts to {OUT}/")
