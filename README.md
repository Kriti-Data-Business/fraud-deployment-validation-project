# Two Hats: Building a Fraud Model, Then Independently Validating It

**What happens when the model developer and the model validator are forced to be the same person — with a wall between them?**

This project plays both roles in a bank's model lifecycle, honestly and in sequence:

- 🎩 **Hat 1 — Line 1 (this repo, Part A):** build a production-shaped **application-fraud detection model** — entity-linkage features, imbalance-aware training, calibrated scores, case-prioritisation metrics.
- 🎩 **Hat 2 — Line 2 (Part B):** take the developer hat off and produce a full **independent validation package** of that same model — conceptual soundness challenge, data-quality assessment, explainability review, challenger benchmarking, stability/drift testing, and a risk-rated findings log — structured the way bank Model Risk Management teams write validation reports (SR 11-7 → SR 26-2 lineage, APRA CPG 235 framing).
- 🤖 **Part C:** add an **LLM fraud-triage assistant** and validate *it* — evaluation harness, guardrails, red-team failure-mode testing, and a continuous-monitoring design for exactly the class of system that current model-risk guidance (SR 26-2, Apr 2026) leaves out of formal scope.

Why this shape: banks run on the three-lines-of-defence model, but almost every portfolio project only ever wears Hat 1. The interesting craft — and the hiring gap — is in Hat 2: proving whether a model is *fit for purpose*, not just performant. Building both sides against the same model, on data with known ground truth, makes the validation checkable rather than performative.

## Part A: the fraud model

### Data

Synthetic application stream (~80k applications over 180 days, ~2.5% fraud) with two engineered fraud patterns:

- **Ring / synthetic-identity fraud** (60% of fraud): bursts of 5–25 applications sharing small pools of devices, addresses and phones; thin-file young applicants, inflated incomes, night-hour skew.
- **Opportunistic first-party fraud** (40%): singleton applications with overstated income relative to the requested amount.

Synthetic-first is a deliberate choice, not a shortcut: **known ground truth is what makes Part B's validation provable** — the validator's findings can be checked against how the data was actually generated. The pipeline is dataset-agnostic; swapping in [IEEE-CIS Fraud Detection](https://www.kaggle.com/c/ieee-fraud-detection) (requires accepting Kaggle competition rules) is a documented extension.

### Approach

| Decision | Choice | Why |
|---|---|---|
| Features | Point-in-time only: trailing 7/30-day entity velocity, expanding entity-degree (emails-per-device etc.), affordability ratios, behavioural | No temporal/target leakage — the property Part B explicitly tests |
| Split | Time-ordered 70/15/15, never random | Fraud drifts; random splits leak future ring members into training |
| Model | LightGBM, `scale_pos_weight` for imbalance (no resampling) | Keeps probability scale interpretable; avoids oversampling artefacts validators flag |
| Calibration | Isotonic, fitted on the validation fold | Scores feed prioritisation thresholds, so they must behave like probabilities |
| Metrics | **PR-AUC, precision@k%, recall@1% FPR, Brier** — ROC-AUC reported but never headlined | At 2.5% prevalence ROC-AUC flatters; investigators review a top-k% queue |

### Results (out-of-time test fold)

12,000 applications, prevalence 2.03% (`python run_pipeline.py`, seed 42):

| Metric | Value | Reading |
|---|---|---|
| PR-AUC | **0.873** | Headline discrimination at 2% prevalence |
| Precision @ top 1% | **0.992** | An investigator queue of the top 1% is ~99% actual fraud |
| Recall @ 1% FPR | **0.848** | 85% of fraud caught within a 1% false-alarm budget |
| Brier score | **0.0045** | Calibrated scores usable as probabilities |
| ROC-AUC | 0.990 | Reported for completeness — flatters at this prevalence |
| Top-1% capture: ring fraud | 90.0% | Entity-linkage features do their job |
| Top-1% capture: opportunistic | **57.5%** | ⚠️ Known segment weakness — flagged for the Part B findings log |

That last row is the point of this project: a headline PR-AUC of 0.87 looks shippable, but segment analysis shows the model leans on network signals and under-detects singleton fraud. A validation report exists to surface exactly this.

Reproduce with `python run_pipeline.py` — metrics land in `reports/metrics.json`, plots in `reports/figures/` (PR curve, calibration curve, feature importances).

### Quickstart

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python run_pipeline.py            # full pipeline, ~1 min
```

## Part B: the independent validation

`python run_validation.py` re-derives the champion and runs the full check suite — challenger benchmarking, feature-family ablation, income-perturbation stress, CleanLab label-quality scan, PSI drift analysis, SHAP review, segment analysis. The written outcome is **[docs/validation_report.md](docs/validation_report.md)**: 8 risk-rated findings and a *fit-for-purpose-with-conditions* verdict.

What the validation caught that Part A's "shippable" metrics hid:

- **F1 (High):** a simpler challenger *beats* the champion — PR-AUC 0.887 vs 0.873. The complexity was unearned.
- **F2 (High):** ablation shows the "multi-factor" model is secretly a single-factor affordability model (PR-AUC collapses 0.873 → 0.466 without ratio features; removing all entity-linkage features costs 0.004).
- **F3 (High):** understating self-reported income by 20% turns over **92% of the top-1% investigation queue** — the dominant feature is applicant-manipulable.
- **F4–F8:** segment blind spots (broker channel, opportunistic fraud), PSI alarm-fatigue tension (9 features breach drift thresholds while the score is stable), age-proxy fairness risk, label-scanner usage guidance, explanation legibility.

The research-grounded failure-scenario catalog behind these checks: [docs/validation_scenarios.md](docs/validation_scenarios.md).

## Repo map

```
run_pipeline.py            # end-to-end Part A entrypoint
src/twohats/synth_data.py  # fraud-pattern data generator (ground truth by construction)
src/twohats/features.py    # point-in-time feature engineering
src/twohats/train.py       # time-split, imbalance-aware LightGBM, calibration
src/twohats/evaluate.py    # fraud-ops metrics + report artefacts
docs/validation_framework.md  # Part B: the validation methodology (SR 26-2 / APRA-aligned)
docs/PROJECT_PLAN.md       # roadmap and phase gates
```

## Roadmap

- [x] **Part A** — fraud model v1: pipeline, calibrated model, evaluation report
- [x] **Week 1 artifact** — validation framework doc ([docs/validation_framework.md](docs/validation_framework.md))
- [x] **Part B** — independent validation package: model card, CleanLab pass, SHAP review, challenger benchmark, ablation + perturbation stress, PSI drift, risk-rated findings log ([validation report](docs/validation_report.md))
- [ ] **Part C** — LLM fraud-triage assistant + its validation: DeepEval suite, guardrails, red-team failure modes, continuous-monitoring design
- [ ] Stretch: GNN fraud-ring detection on the entity graph

## Author

Kriti Yadav — [GitHub](https://github.com/Kriti-Data-Business) · [Portfolio](https://kriti-data-business.github.io/portfolio-data-science/) · [LinkedIn](https://www.linkedin.com/in/kriti-yadav-data-science/) · [Medium](https://medium.com/@datasciencebusiness)
