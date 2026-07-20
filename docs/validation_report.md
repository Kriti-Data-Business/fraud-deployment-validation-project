# Independent Validation Report

**Model:** Application Fraud Triage Model v1 (calibrated LightGBM) — see [model_card.md](model_card.md)
**Validation date:** 20 July 2026 · **Methodology:** [validation_framework.md](validation_framework.md) (SR 11-7 → SR 26-2 lineage, APRA CPG 235/220 touchpoints)
**Raw evidence:** `reports/validation/validation_results.json`, reproducible via `python run_validation.py`

---

## Verdict: **Fit for purpose — with conditions**

The model is approved for its stated triage-only use: discrimination is strong (PR-AUC 0.873, precision@1% 0.992), calibration is excellent (Brier 0.0045), and the leakage review passed. Approval is **conditional** on remediation of findings F1–F3 before any scope expansion, and on the monitoring plan in §5. Had the intended use been auto-decline rather than human-reviewed triage, findings F2/F3/F6 would have made the verdict **not fit**.

---

## 1. Conceptual soundness

**Passed:** point-in-time feature discipline verified (trailing windows exclude the current application; expanding degree counts use only history). Time-ordered split prevents ring members leaking across folds. Imbalance handled by weighting, not resampling — no synthetic-oversampling artefacts, probability scale preserved.

**Failed — the model's effective theory contradicts its stated theory (→ F2).** The design document claims entity-linkage signals are a core detection mechanism alongside affordability. The evidence says otherwise:

| Ablation variant | PR-AUC | Δ vs full |
|---|---|---|
| Full model | 0.873 | — |
| **Without entity-linkage family** (11 features) | **0.869** | **−0.004** |
| **Without affordability family** (4 features) | **0.466** | **−0.407** |

Removing every entity-linkage feature costs almost nothing; removing affordability halves performance. This is effectively a single-factor model wearing a multi-factor design document. SHAP confirms: `amount_income_ratio` (1.06 mean |SHAP|) and `employment_age_ratio` (0.72) dominate all linkage features.

## 2. Outcome analysis

### 2.1 Benchmarking — the champion does not earn its complexity (→ F1)

| Model | PR-AUC | Precision@1% | Recall@1% FPR |
|---|---|---|---|
| Champion (calibrated LightGBM, 2000 trees, tuned) | 0.873 | 0.992 | 0.848 |
| **Challenger: shallow GBM (300 trees, 15 leaves)** | **0.887** | 1.000 | 0.853 |
| Challenger: logistic regression | 0.857 | 1.000 | 0.828 |

A materially simpler challenger **outperforms the champion**, and an interpretable logistic regression lands within 0.02 PR-AUC. The champion's complexity is unjustified; a simpler model would be easier to explain, monitor and defend.

### 2.2 Segment performance — headline metrics hide a blind spot (→ F4)

| Segment | Evidence |
|---|---|
| Ring fraud | 90.0% captured in top-1% queue |
| **Opportunistic fraud** | **57.5% captured** — the model under-detects singleton fraud |
| Channel: online | PR-AUC 0.897 |
| **Channel: broker** | **PR-AUC 0.777** — weakest channel, no channel-level monitoring proposed |

### 2.3 Sensitivity — a self-reported field steers the queue (→ F3)

Perturbing stated income by **−20%** (i.e., the applicant declares less) causes **91.7% turnover of the top-1% investigation queue** and a mean absolute score shift of 0.122. Income is self-reported and unverified in this design; the model's dominant feature is directly manipulable by the applicant. (+20% income: 30% queue turnover.)

### 2.4 Label quality (validator-side, CleanLab confident learning)

25,000 training rows scanned; 83 flagged (0.33%). Ground-truth labels are correct by construction here, so this measures the **scanner's own false-positive behaviour**: flags concentrate in opportunistic-fraud cases (41/83) — the genuinely ambiguous boundary region — not random noise. Production guidance: treat confident-learning flags as *hard-case triage*, never as automatic label edits (→ F7).

### 2.5 Calibration

Reliability curve tracks the diagonal across quantile bins; Brier 0.0045. Isotonic step retained. **No finding.**

## 3. Explainability review

Global attribution structure is coherent and case-level explanations for the top-3 queue applications cite plausible drivers (a ring case is driven by `address_id_apps_7d`, `phone_apps_7d`; opportunistic cases by affordability ratios). Two caveats: (a) one top case shows a negative `devices_per_address` contribution that an investigator would find counter-intuitive — explanation templates must translate SHAP values, not expose them raw; (b) `age` and its derivative `employment_age_ratio` are high-attribution features (→ F6).

## 4. Stability / drift

| Check | Result | Reading |
|---|---|---|
| Score PSI (train → out-of-time) | 0.008 — stable | Output distribution unchanged |
| Feature PSI | **9 of 24 features ≥ 0.1**; `devices_per_address` 0.85, `address_id_apps_*` 0.83, `phones_per_device` 0.46 | Entity-velocity features are inherently bursty — ring activity makes their distributions non-stationary **by design** |

The instructive tension (→ F5): naive per-feature PSI alerting at the industry 0.1/0.25 heuristics would fire **continuously** on velocity features while the model performs perfectly. Monitoring thresholds must be governance-calibrated per feature family, or drift alerting becomes noise that teams learn to ignore — the worst possible monitoring outcome.

## 5. Ongoing monitoring plan (condition of approval)

1. **Score PSI weekly** (threshold 0.1 investigate / 0.25 escalate) — primary drift signal, since feature PSI is structurally noisy here.
2. **Rolling precision@1% monthly** on confirmed outcomes, with the label-delay blind period explicitly annotated (labels lag 1–6 months in production fraud).
3. **Segment dashboards**: fraud-type and channel capture rates (broker channel priority).
4. **Queue-composition alert**: >50% week-on-week turnover of the top-1% queue without a policy change triggers review (detects both drift and adversarial probing).
5. **Annual revalidation**, or immediately on: retraining, new channel, or entity-resolution vendor change (APRA CPS 230 service-provider clause).

## 6. Findings log

| ID | Severity | Finding | Recommendation | Status |
|---|---|---|---|---|
| **F1** | **High** | Champion outperformed by a simpler challenger (PR-AUC 0.887 vs 0.873); complexity unjustified | Re-select or re-tune; default to the simplest model within 1 SE of best | Open |
| **F2** | **High** | Effective single-factor dependence on affordability ratios (ablation −0.407 PR-AUC); stated multi-factor theory not realised | Rebalance feature contribution (e.g., monotonic constraints, per-family regularisation) or deploy a linkage-rule overlay for ring detection | Open |
| **F3** | **High** | Top-1% queue is 92%-steerable by understating self-reported income | Add income verification (bank-statement/bureau) for high-score applications; cap ratio features; adversarial-robustness test in next cycle | Open |
| **F4** | Medium | Segment blind spots: opportunistic capture 57.5% vs ring 90%; broker channel PR-AUC 0.777 vs online 0.897 | Segment-aware thresholding or dedicated opportunistic-fraud features; channel-level monitoring (§5.3) | Open |
| **F5** | Medium | Industry-heuristic PSI thresholds would generate continuous false drift alarms on structurally bursty velocity features | Governance-calibrated, per-family thresholds; score-PSI as primary signal (§5.1) | Accepted into monitoring plan |
| **F6** | Medium | `age` used directly and via `employment_age_ratio` (high SHAP attribution) — age-discrimination / proxy risk under AU law if scores ever influence decline decisions | Fairness review before any scope expansion beyond triage; test age-blind variant | Open |
| **F7** | Low | Label-noise scanner flags 0.33% on provably clean labels, concentrated in hard cases | Use confident-learning flags for investigator triage only, never automated relabelling | Closed (guidance) |
| **F8** | Low | Local SHAP outputs not investigator-legible without translation | Explanation templates in the case-management UI | Open |

**Positive assurance:** leakage review passed · calibration excellent · metrics correctly chosen for prevalence (PR-AUC/precision@k headline, ROC-AUC demoted) · reproducible pipeline (single-command, seeded).

---

*Prepared under the Two Hats protocol: the validator role had no authority to edit the model, only to test it. Every number above is reproducible from the repo.*
