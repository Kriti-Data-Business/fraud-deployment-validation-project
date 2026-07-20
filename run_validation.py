"""Part B entrypoint: independent validation of the Part A champion model.

Reproduces the champion deterministically, then executes the quantitative
validation checks from docs/validation_framework.md and writes:

    reports/validation/validation_results.json   (all raw check outputs)
    reports/validation/figures/*.png             (SHAP summary etc.)

The written validation report (findings log, ratings, verdict) lives in
docs/validation_report.md and cites these artefacts.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

import numpy as np

from twohats.evaluate import compute_metrics, precision_at_k
from twohats.features import build_features
from twohats.synth_data import generate
from twohats.train import predict_calibrated, time_split, train_model
from twohats.validation.challengers import run_challengers
from twohats.validation.drift import feature_psi_table, psi, rate_psi
from twohats.validation.label_quality import label_issue_scan
from twohats.validation.shap_review import shap_review
from twohats.validation.stress import family_ablation, income_perturbation

OUT = Path("reports/validation")


def main() -> None:
    t0 = time.time()
    results: dict = {}

    print("[1/7] Reproducing champion (same seed/split as Part A)...")
    df = build_features(generate())
    train, valid, test = time_split(df)
    model, calibrator, cols = train_model(train, valid)
    scores = predict_calibrated(model, calibrator, test, cols)
    y_test = test["is_fraud"].to_numpy()
    results["champion_metrics"] = compute_metrics(y_test, scores)

    print("[2/7] Segment analysis (fraud type / channel / amount band)...")
    seg = {}
    k_mask = scores >= np.sort(scores)[::-1][max(1, int(len(scores) * 0.01)) - 1]
    for name, mask in [
        ("ring", test["fraud_type"].to_numpy() == "ring"),
        ("opportunistic", test["fraud_type"].to_numpy() == "opportunistic"),
        ("channel=online", (test["channel"] == "online").to_numpy()),
        ("channel=broker", (test["channel"] == "broker").to_numpy()),
        ("amount>=100k", (test["requested_amount"] >= 100_000).to_numpy()),
    ]:
        m = mask & (y_test == 1) if name in ("ring", "opportunistic") else mask
        frauds = y_test[mask] if name.startswith(("channel", "amount")) else None
        seg[name] = {
            "n": int(mask.sum()),
            "top1pct_capture": float(k_mask[m].mean()) if m.any() else None,
        }
        if frauds is not None and mask.sum() > 50:
            seg[name]["pr_auc"] = float(
                compute_metrics(y_test[mask], scores[mask])["pr_auc"]
            ) if y_test[mask].sum() > 5 else None
    results["segments"] = seg

    print("[3/7] Label-quality scan (CleanLab confident learning)...")
    results["label_quality"] = label_issue_scan(train, cols)

    print("[4/7] Challenger benchmarking...")
    results["challengers"] = run_challengers(train, test, cols, scores).to_dict(orient="records")

    print("[5/7] Sensitivity & ablation stress tests...")
    results["income_perturbation"] = income_perturbation(model, calibrator, test, cols)
    results["family_ablation"] = family_ablation(train, valid, test, cols).to_dict(orient="records")

    print("[6/7] Drift analysis (PSI, train window vs out-of-time window)...")
    psi_table = feature_psi_table(train, test, cols)
    train_scores = predict_calibrated(model, calibrator, train, cols)
    score_psi = psi(train_scores, scores)
    results["drift"] = {
        "score_psi": {"psi": round(score_psi, 4), "rating": rate_psi(score_psi)},
        "feature_psi_top10": psi_table.head(10).to_dict(orient="records"),
        "n_features_flagged": int((psi_table["psi"] >= 0.1).sum()),
    }

    print("[7/7] SHAP explainability review...")
    results["shap"] = shap_review(model, test, cols, scores, OUT / "figures")

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "validation_results.json").write_text(json.dumps(results, indent=2, default=str))

    print(f"\nDone in {time.time() - t0:.1f}s. Key outputs:")
    print(f"  challengers: {json.dumps(results['challengers'], indent=2)}")
    print(f"  ablation:    {json.dumps(results['family_ablation'], indent=2)}")
    print(f"  perturbation:{json.dumps(results['income_perturbation'], indent=2)}")
    print(f"  label scan:  {json.dumps(results['label_quality'], indent=2)}")
    print(f"  drift:       score PSI {results['drift']['score_psi']}, "
          f"{results['drift']['n_features_flagged']} features >= 0.1; "
          f"top: {json.dumps(results['drift']['feature_psi_top10'][:4], indent=2)}")
    print(f"  segments:    {json.dumps(results['segments'], indent=2)}")
    print(f"  shap top10:  {json.dumps(results['shap']['global_top10_mean_abs_shap'], indent=2)}")
    print(f"  locals:      {json.dumps(results['shap']['top_queue_local_explanations'], indent=2)}")
    print(f"Full results: {OUT}/validation_results.json")


if __name__ == "__main__":
    main()
