# Case-Scenario Catalog: How Fraud Models Fail Silently

*Research-grounded failure scenarios for fraud/credit ML systems, each mapped to the validation check that catches it and to what this project's own validation actually found. Sources at the end.*

| # | Scenario | How it happens | The check that catches it | This project's evidence |
|---|---|---|---|---|
| 1 | **Resampling / temporal leakage** | SMOTE or scaling applied before the train-test split; random splits leak future fraud-ring members into training | Leakage review: point-in-time feature audit, time-ordered splits, pipeline-order inspection | Passed — designed out from day one, verified in §1 of the validation report |
| 2 | **The delayed-label blind period** | Fraud confirmation arrives 1–6 months after prediction; the model silently decays while dashboards show stale "performance" | Monitoring design that annotates label lag; early proxy signals (disputes, soft labels — cf. Stripe's ML Flywheel) | Monitoring plan §5.2 requires blind-period annotation |
| 3 | **Selective-labels feedback loop** | Only investigated cases get labels; the model retrains on its own past decisions and calcifies its blind spots | Validator-side label-quality scan + sampling review queue below threshold | CleanLab scan run; flags shown to concentrate in ambiguous cases (F7) |
| 4 | **Adversarial adaptation** | Fraudsters probe thresholds and adapt — fraud detection is a cat-and-mouse game; a single-training model is vulnerable to even simple adaptation | Sensitivity/perturbation testing on manipulable inputs | **Caught: −20% stated income turns over 92% of the investigation queue (F3)** |
| 5 | **Hidden single point of failure** | One feature family carries the model; a data outage or evasion of that family collapses detection | Feature-family ablation | **Caught: removing affordability ratios halves PR-AUC, 0.873 → 0.466 (F2)** |
| 6 | **Unearned complexity** | The tuned champion ships because it's the champion; nobody checks whether a simpler model matches it | Independent challenger benchmark (interpretable floor + alternative config) | **Caught: shallow GBM challenger beats champion, 0.887 vs 0.873 (F1)** |
| 7 | **Segment blind spots behind headline metrics** | Aggregate PR-AUC looks shippable while a fraud type or channel is under-detected | Segment-level outcome analysis by fraud type, channel, amount band | **Caught: opportunistic capture 57.5% vs ring 90%; broker PR-AUC 0.777 (F4)** |
| 8 | **Drift-alarm fatigue** | Industry PSI heuristics (0.1/0.25) applied uniformly; structurally bursty features alarm continuously until the team ignores all alerts | Governance-calibrated per-family thresholds; score-PSI as primary signal | **Caught: 9/24 features breach PSI 0.1 while score PSI is 0.008 and performance is stable (F5)** |
| 9 | **Proxy discrimination** | Age/postcode enter as innocent features; scores later gate decisions and become indirect discrimination | Explainability review of protected-attribute attribution + fairness testing gate before scope expansion | **Caught: age and employment_age_ratio carry high SHAP attribution (F6)** |
| 10 | **Miscalibrated scores misused as probabilities** | Rank-good, probability-bad scores feed risk-appetite thresholds and expected-loss maths | Reliability curves + Brier, on the out-of-time fold | Passed — isotonic calibration, Brier 0.0045 |

## The meta-lesson

Seven of these ten scenarios are invisible to the standard portfolio-project workflow (train → test → report AUC → ship). They only surface when someone whose job is *not* shipping the model goes looking. That is the entire argument for independent validation — and why regulators have required it since SR 11-7 (2011) and still do under SR 26-2 (2026).

## Sources

- [Impact of sampling techniques and data leakage on XGBoost fraud detection (arXiv)](https://arxiv.org/abs/2412.07437)
- [Proactive detection of model degradation with delayed labels (ECML PKDD)](https://ecmlpkdd-storage.s3.eu-central-1.amazonaws.com/preprints/2025/ads/preprint_ecml_pkdd_2025_ads_315.pdf) — incl. Stripe ML Flywheel & Adyen ghost/challenger/principal architectures
- [Adversarial learning in real-world fraud detection (ACM/ResearchGate)](https://www.researchgate.net/publication/373761854_Adversarial_Learning_in_Real-World_Fraud_Detection_Challenges_and_Perspectives)
- [FPR estimation under class-conditional label noise (arXiv)](https://arxiv.org/pdf/2308.02695)
- [Fair classification with partial feedback / selective labels (arXiv)](https://arxiv.org/pdf/2402.11338)
- [Label-free detection of governance evidence degradation in risk decision systems (arXiv)](https://arxiv.org/pdf/2604.17836) — PSI heuristics discussion
- [Model drift in ML (Aerospike)](https://aerospike.com/blog/model-drift-machine-learning/)
