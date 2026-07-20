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

- [ ] Model card (intended use, non-uses, assumptions register)
- [ ] Conceptual-soundness challenge incl. leakage review
- [ ] Validator-side data quality: CleanLab label-noise scan, independent EDA
- [ ] SHAP explainability review (global + case-level for top-queue applications)
- [ ] Challenger models: regularised logistic regression + alternative GBM; champion-vs-challenger table
- [ ] Sensitivity/stress: input perturbation, feature-family ablation
- [ ] PSI/CSI drift analysis across time windows
- [ ] Risk-rated findings log (starting with the known opportunistic-segment weakness) + fit-for-purpose verdict
- [ ] **Validation report PDF** — the flagship portfolio artefact

## Gate 3 — LLM triage assistant + its validation *(target ~13 Sep 2026)*

- [ ] LLM case-summary/triage assistant over flagged applications
- [ ] DeepEval suite: faithfulness, groundedness, format compliance
- [ ] Red-team set: prompt injection via application text, contradiction traps; measured failure rates
- [ ] Guardrails layer + tests of the guardrails themselves
- [ ] Continuous-monitoring + kill-switch/fallback design doc
- [ ] Medium article published; portfolio entry updated

## Stretch

- [ ] GNN fraud-ring detection on the entity graph; compare against velocity features
- [ ] Swap in IEEE-CIS dataset (requires Kaggle rules acceptance) and rerun the full validation
