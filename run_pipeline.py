"""End-to-end Part A pipeline: data -> features -> model -> evaluation report.

Usage:
    python run_pipeline.py                  # 80k synthetic applications
    python run_pipeline.py --n-apps 200000  # bigger run
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from twohats.evaluate import compute_metrics, save_report
from twohats.features import build_features
from twohats.synth_data import generate
from twohats.train import gain_importances, predict_calibrated, time_split, train_model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-apps", type=int, default=80_000)
    parser.add_argument("--fraud-rate", type=float, default=0.025)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", default="reports")
    args = parser.parse_args()

    t0 = time.time()
    print(f"[1/4] Generating {args.n_apps:,} synthetic applications "
          f"(fraud rate {args.fraud_rate:.1%}, seed {args.seed})...")
    df = generate(n_apps=args.n_apps, fraud_rate=args.fraud_rate, seed=args.seed)
    Path("data").mkdir(exist_ok=True)
    df.to_csv("data/applications_synthetic.csv.gz", index=False, compression="gzip")

    print("[2/4] Building point-in-time features...")
    df = build_features(df)

    print("[3/4] Training LightGBM (time-ordered 70/15/15 split, calibrated)...")
    train, valid, test = time_split(df)
    model, calibrator, cols = train_model(train, valid)

    print("[4/4] Evaluating on the out-of-time test fold...")
    scores = predict_calibrated(model, calibrator, test, cols)
    y_test = test["is_fraud"].to_numpy()
    metrics = compute_metrics(y_test, scores)
    metrics["train_rows"] = len(train)
    metrics["best_iteration"] = int(model.best_iteration_ or model.n_estimators)
    save_report(metrics, y_test, scores, gain_importances(model, cols), args.out)

    # Per-fraud-type recall in the top 1% — are we catching rings, or only
    # the easy opportunistic cases? (Feeds the Part B outcome analysis.)
    top1 = scores >= sorted(scores, reverse=True)[max(1, int(len(scores) * 0.01)) - 1]
    by_type = {
        ft: float(top1[test["fraud_type"].to_numpy() == ft].mean())
        for ft in ("ring", "opportunistic")
    }
    metrics["top1pct_capture_by_fraud_type"] = by_type
    (Path(args.out) / "metrics.json").write_text(json.dumps(metrics, indent=2))

    print(f"\nDone in {time.time() - t0:.1f}s. Test fold ({metrics['n']:,} apps, "
          f"prevalence {metrics['prevalence']:.2%}):")
    for k in ("pr_auc", "roc_auc", "precision_at_1pct", "precision_at_5pct",
              "recall_at_1pct_fpr", "brier_score"):
        print(f"  {k:22s} {metrics[k]:.4f}")
    print(f"  top-1% capture — ring: {by_type['ring']:.1%}, "
          f"opportunistic: {by_type['opportunistic']:.1%}")
    print(f"Artefacts: {args.out}/metrics.json, {args.out}/figures/*.png")


if __name__ == "__main__":
    main()
