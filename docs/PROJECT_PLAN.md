# Project plan & phase gates

Target roles this project speaks to: AI/model validation in Model Risk Management (Line 2) and hands-on fraud data science (Line 1) in regulated financial services.

## Gate 1 — Fraud model v1 + validation literacy *(target ~2 Aug 2026)*

- [x] Repo scaffolded, reproducible pipeline (`run_pipeline.py`)
- [x] Synthetic application-fraud generator with ring + opportunistic patterns (known ground truth)
- [x] Point-in-time features: entity velocity, expanding entity-degree, affordability, behavioural
- [x] Time-ordered split, imbalance-aware LightGBM, isotonic calibration
- [x] Fraud-ops evaluation: PR-AUC 0.873, precision@1% 0.992, recall@1%FPR 0.848, segment capture by fraud type
- [x] Validation framework doc ([validation_framework.md](validation_framework.md)) — SR 11-7 → SR 26-2 lineage, APRA touchpoints, three-pillar methodology
- [ ] Read-through of primary sources: SR 26-2 summary, CPG 235, Edinburgh GenAI-MRM paper (personal study, ~4h)
- [ ] Push to GitHub, add repo to portfolio site

## Gate 2 — Independent validation package *(target ~23 Aug 2026)*

- [x] Model card (intended use, non-uses, assumptions register) — [model_card.md](model_card.md)
- [x] Conceptual-soundness challenge incl. leakage review
- [x] Validator-side data quality: CleanLab label-noise scan (25k rows, scanner-FPR insight)
- [x] SHAP explainability review (global + case-level for top-queue applications)
- [x] Challenger models: logistic regression + shallow GBM — **challenger beat champion (F1)**
- [x] Sensitivity/stress: income ±20% perturbation, feature-family ablation — **F2, F3**
- [x] PSI drift analysis across time windows — **alarm-fatigue finding (F5)**
- [x] Risk-rated findings log (8 findings) + verdict: fit for purpose with conditions — [validation_report.md](validation_report.md)
- [x] Failure-scenario catalog from web research — [validation_scenarios.md](validation_scenarios.md)
- [ ] LinkedIn carousel PDFs (2) — story carousel + scenario-checklist carousel

## Gate 3 — LLM triage assistant + its validation *(target ~13 Sep 2026)*

- [x] Case-summary/triage assistant contract over flagged applications (baseline + LLM adapter; live model not yet run)
- [x] Deterministic evaluators: groundedness, citation, format compliance (DeepEval not used; a model-graded judge is still to add)
- [x] Red-team set: prompt injection via applicant note, contradiction traps; measured failure rates (stand-in assistants)
- [x] Guardrails layer + tests of the guardrails themselves
- [x] Continuous-monitoring + kill-switch/fallback design doc (design only)
- [ ] Run a live LLM through the harness; add semantic judge (C2)
- [ ] Medium article published; portfolio entry updated

## Stretch

- [ ] GNN fraud-ring detection on the entity graph; compare against velocity features
- [ ] Swap in IEEE-CIS dataset (requires Kaggle rules acceptance) and rerun the full validation
