# Independent Validation Framework

*Week-1 literacy artifact for the Two Hats project: the methodology Part B will execute, grounded in current model-risk regulation. Doubles as the outline for the companion Medium article.*

## 1. The regulatory frame (what changed in 2026)

- **SR 11-7 (Fed, 2011)** was the canonical model-risk guidance for 15 years: model inventory, validation independent of development, ongoing monitoring, board-level governance. Its core definition of model risk — *the potential for adverse consequences from decisions based on incorrect or misused model outputs* — still underpins everything.
- **SR 26-2 (Fed/OCC/FDIC, April 2026)** rescinded SR 11-7 and replaced it with a more risk-based, principles-driven framework. Two things matter for this project:
  1. Validation intensity should be **proportionate to model materiality** — a fraud model gating credit decisions is high-materiality.
  2. **Generative and agentic AI are excluded from formal scope** — regulators judged them too novel and fast-moving. Every bank deploying LLM systems is therefore improvising validation practice right now. Part C lives in exactly this gap, and examiners are already probing it (unauthorized data use, kill-switch capability, vendor AI governance).
- **Australian frame (APRA):** no direct SR 11-7 equivalent, but the same expectations arrive via **CPG 235** (data risk: validation of data fitness-for-use, controls, lineage), **CPS/CPG 220** (risk management framework, independent review), and **CPS 230** (operational risk — models as critical operations, including service-provider/vendor models). Australian MRM teams (e.g. at Macquarie) blend SR 11-7-style validation craft with APRA's principles-based expectations.

**Design consequence for Part B:** the validation report follows the SR 11-7/26-2 skeleton because that's the shared craft vocabulary, annotated with APRA touchpoints because that's the jurisdiction.

## 2. The three pillars of validation (and how Part B implements each)

### Pillar 1 — Conceptual soundness
*Is the design defensible for the stated purpose, before looking at performance?*

| Check | Part B implementation |
|---|---|
| Purpose & use documented | Model card: intended use (application-fraud triage), explicit non-uses (not a credit-worthiness score) |
| Theory of the features | Challenge each feature family: is entity-velocity a causal fraud signal or a data artefact? |
| Data representativeness | Population definition, class-imbalance handling rationale, known-limitations register |
| Assumptions register | Time-stationarity, entity-resolution quality, label completeness (fraud labels are always partial in reality) |
| Leakage review | Verify every feature is point-in-time; attempt to construct a leaking feature and show the pipeline blocks it |

### Pillar 2 — Outcome analysis
*Does it work, measured the way the business uses it?*

| Check | Part B implementation |
|---|---|
| Discrimination | PR-AUC (headline), ROC-AUC (context only at 2.5% prevalence) |
| Operational value | Precision@0.5/1/5% (investigator queue quality), recall@1% FPR |
| Calibration | Reliability curve + Brier; isotonic-calibration effectiveness check |
| Segment performance | By fraud type (ring vs opportunistic), channel, amount band — a model that only catches easy fraud "performs well" and fails its purpose |
| Data quality | CleanLab label-noise scan; validator-side EDA independent of the developer's |
| **Benchmarking (challenger)** | Independent challenger models: regularised logistic regression (interpretable floor) + alternative GBM config; the champion must earn its complexity |
| Sensitivity & stress | Perturb key inputs (income ±20%, entity-linkage removal) and measure score stability; ablate feature families to find single-points-of-failure |

### Pillar 3 — Ongoing monitoring
*Will we know when it stops working?*

| Check | Part B implementation |
|---|---|
| Input drift | PSI/CSI per feature, train vs out-of-time windows; alert thresholds (PSI > 0.1 investigate, > 0.25 act) |
| Score drift | Score-distribution PSI over time |
| Performance decay | Rolling precision@k on labelled outcomes; label-lag caveat documented |
| Override/usage monitoring | Design only (no live users): what would be tracked and when review triggers |

### Findings log format

Every finding gets: ID · severity (**High / Medium / Low**, by impact on fit-for-purpose) · evidence · recommendation · developer response. Validation concludes with one of: *fit for purpose / fit with conditions / not fit* — mirroring real approval decisions, with conditions listed.

## 3. Validating the LLM component (Part C preview)

Traditional validation assumes a fixed input→output mapping evaluated pre-deployment. LLM triage assistants break that: outputs are generative, failure modes are behavioural (hallucinated case facts, prompt injection via application free-text, unfaithful summaries, PII leakage), and pre-deployment testing alone is insufficient. Part C therefore shifts weight to:

1. **Capability evaluation** — DeepEval suite: faithfulness of case summaries to source records, groundedness, format compliance.
2. **Failure-mode red-teaming** — adversarial test set: injection attempts embedded in application text, contradictory records, edge-case prompts; measured jailbreak/leak rates.
3. **Guardrails as controls** — input/output validators (PII, topic, schema) with the guardrail layer itself tested, not assumed.
4. **Continuous monitoring design** — sampled human review, drift alerts on output characteristics, kill-switch/fallback path (queue reverts to score-only prioritisation) — directly answering the 2026 examiner focus areas.

## 4. Medium article outline ("I built a fraud model, then validated it the way a bank would")

1. Hook: every DS portfolio has a fraud model; nobody shows you the document that decides whether it ships.
2. The three-lines-of-defence model, and why validators are the interesting job market nobody talks about.
3. What SR 26-2 changed in April 2026 — and the GenAI gap.
4. Wearing Hat 1 honestly (leakage-free features, calibration, top-k metrics).
5. Wearing Hat 2: the findings log — what my own model got wrong.
6. What this taught me about AI validation as a career direction.

## Sources

- [SR 26-2 overview — AegisAI](https://www.aegisaicompliance.com/resources/sr-26-2-model-risk-management-ai-governance) · [Databricks: MRM in 2026](https://www.databricks.com/blog/model-risk-management-2026-bankers-guide-revised-interagency-guidance) · [CRA Insights: SR 26-2 in the era of AI](https://media.crai.com/wp-content/uploads/2026/07/09165122/FE-Insights-MRM-Guidance-SR-26-2-In-the-Era-of-AI-July2026.pdf)
- [Examiner focus on GenAI outside SR 26-2 scope](https://www.techtimes.com/articles/318340/20260613/bank-ai-oversight-expands-every-exam-generative-ai-bypasses-sr-26-2-kill-switch-gap-grows.htm)
- [APRA CPG 235](https://www.apra.gov.au/practice-guides/cpg-235) · [APRA CPG 220](https://handbook.apra.gov.au/ppg/cpg-220)
- [Effective Generative AI Model Risk Management (Edinburgh CRC)](https://www.crc.business-school.ed.ac.uk/sites/crc/files/2025-11/Effective-Generative-AI-Model-Risk-Management-paper.pdf)
- [SR 26-2-compatible GenAI risk framework (arXiv)](https://arxiv.org/pdf/2607.04103)
