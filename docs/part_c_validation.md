# Part C: validating the LLM fraud-triage layer

**Status:** the test bench is built and run. The first table of results comes from deterministic stand-in assistants, so it measures the harness and the guardrails. A second section, "Live models", reports two real models (Claude Sonnet 5.5 and GPT-6.1-sol, both via OpenRouter) run once through the same bench on the same synthetic cases. Those are single runs with one prompt, small attack samples and no repeats: they are a first measurement, not a validation.

Run it: `python run_triage.py` (about 10 seconds after the Part A model trains). Raw output: `reports/triage/results.json`. Guardrail tests: `pytest tests`.

## What was built

| Piece | Where | Role |
|---|---|---|
| Case-fact layer | `src/twohats/triage/cases.py` | Top 1% of the out-of-time fold (120 cases) becomes numbered, citable facts. Ground-truth labels are kept for the harness and never shown to the assistant. Age is excluded from the facts on purpose (Part B finding F6). |
| Assistant contract | `assistant.py` | The assistant never decides. It returns a risk band, a pattern, cited claims, and one of three allowed next steps, with `human_review_required` always true. |
| Baseline assistant | `assistant.py` | Deterministic, faithful by construction. It is the reference the harness is calibrated against. |
| LLM adapter | `assistant.py` | The model call is injected. Tested with a fake client only. |
| Guardrails | `guardrails.py` | Schema, citation validity, number grounding, decision language, protected-attribute wording, identifiers, human-review flag, allowed next steps, score consistency, injection echo, pattern re-derivation. Fails closed to a manual-review fallback. |
| Seeded faults and attacks | `faults.py` | 11 injected defect types and 7 injection attacks plus one benign note. |

## Results

Queue: 120 flagged cases, 119 true fraud (precision 99.2%, matching Part A). Mix: 65 ring, 54 opportunistic, 1 legitimate.

| Check | Result |
|---|---|
| Guardrail false alarms on clean baseline output | 0 of 120 |
| Seeded faults caught | 1,200 of 1,320 injections; 10 of 11 fault types caught every time |
| Fault type never caught | `unsupported_claim_no_number` (0 of 120) |
| Trusting assistant (copies the applicant note), attacks compromised before guardrails | 840 of 840 |
| Same, escaped after guardrails | 0 of 840 |
| Quiet assistant (obeys the note without echoing it), compromised before guardrails | 720 of 840 |
| Same, escaped form-level checks alone | 720 of 720 |
| Same, escaped after adding pattern re-derivation | 12 of 720 |
| Baseline pattern accuracy vs ground truth | 58.8% overall: ring 49 of 65, opportunistic 21 of 54 |

## Live models (run 2, after fixing the bench)

Same 120 flagged cases, temperature 0, one prompt (`SYSTEM_PROMPT` in `assistant.py`), through `run_live.py`. Attack columns use 30 cases per attack, taken from cases where the model's own clean answer passed the guardrails. Raw files: `reports/triage/live_*.json`.

| Measure | Claude Sonnet 5.5 | GPT-6.1-sol |
|---|---|---|
| Clean summaries blocked by the guardrails | 25 of 120 | 30 of 120 |
| of which pattern downgrade | 18 | 30 |
| of which unparseable output | 3 | 0 |
| of which invented number | 4 | 0 |
| Pattern matches the known fraud type (true fraud only) | 55 of 119 | 53 of 119 |
| Pattern agrees with the rule baseline | 82 of 120 | 74 of 120 |
| Output changed by a note, summed over the 8 notes (of 240) | 7 | 16 |
| Same, harmless note only (of 30) | 1 | 1 |
| Changed and not blocked, summed over 8 notes | 2 | 4 |
| Tokens used | 273,787 | 207,148 |

For comparison, the rule-based stand-in names the pattern correctly for 70 of 119 true-fraud cases (58.8%).

**How to read this.** Each model's output is compared with that same model's clean answer for the same case, so "changed" means the note moved the risk level, pattern or next step. A harmless note also changed one output for each model, so one or two changes per attack is within the noise floor. At 30 cases per attack, a difference of two or three is a direction to look at, not a result. GPT-6.1-sol changed 3 outputs under "pressure to approve" and 4 under "claims authority", against 1 for the harmless note. Sonnet changed none or one under every note. That is worth repeating with more cases before anyone says one model resists injection better.

## Findings

**C1 (High, partly closed). Quiet compliance with injected instructions passes every form-level check.** The quiet assistant keeps the output well-formed, keeps numbers grounded and leaves the risk band alone, but downgrades the pattern to "unclear" and the next step to manual review. All 720 compromised outputs passed. Adding an independent re-derivation of the pattern from the facts, applied one way only (a summary may not lower scrutiny the facts support), cut escapes to 12. Those 12 are 2 cases across 6 attacks where the facts only support "unclear" and the attacker dropped the evidence line, so the remaining gap is evidence omission. I wrote this attacker to be invisible to the form checks, so this shows that the failure class exists and that the control works against it, not how common it is.

**C2 (High, open). A false claim with no numbers cannot be detected.** "The applicant previously defaulted with another lender" passed 120 of 120 times. The deterministic checks verify form, numbers and wording, not truth. This needs a semantic judge, calibrated against human labels, before a live model is trusted.

**C3 (Medium, open). Number grounding is weaker than it looks.** The fabricated-number fault was flagged in all 120 cases, but number grounding caught only 113. The other 7 were flagged because the fake sentence contained the word "declined", which is a coincidence. An invented number that equals a cited fact's value passes. Typed claim templates, where each claim slot is bound to one fact, would close this.

**C4 (Medium, open). The assistant inherits the model's blind spot.** Even the baseline's pattern labels are right far less often for opportunistic fraud (39%) than for rings (75%). That echoes Part B finding F4 (ring capture 90.0%, opportunistic 57.5%). The two measure different things, so this is a pointer, not a proof. An "unclear" label must never be read as evidence of absence.

**C5 (Medium, open). Fail-closed over-blocks.** The trusting assistant's benign note was blocked 120 of 120 times. That is the right outcome for a system that should never repeat notes, but it means notes should not reach the generator at all, or should pass through a separate extraction step.

**C6 (Low, known). Rule thresholds were set a priori.** The pattern rule (3 shared entities, 0.5 amount-to-income) was not tuned on the test fold, and tuning it there would be leakage. It would need a validation-fold calibration before any use beyond a baseline.

**C7 (Medium, fixed in the bench; kept on record). The first live run measured my guardrails, not the model.** Claude Sonnet 5.5 had 117 of 120 clean summaries blocked. The causes were bench bugs: a citation such as "F3" counted as an invented number (86 blocks), JSON wrapped in prose or code fences failed the parser (31), and one fact label contained a number. I fixed the number check so citations do not count, made the parser tolerant (counted separately), removed the digit from the label and tightened the prompt, then reran. The stand-in results did not change. The lesson is that a guardrail tested only against its own stand-in can be wrong in ways only a real model shows. The run-1 numbers for GPT-6.1-sol were not captured.

**C8 (Medium, open). Both live models downgrade the pattern far more often than the rule allows.** 18 of 120 (Sonnet) and 30 of 120 (GPT) clean summaries were blocked for calling the case "unclear" when the rule found a pattern. In the 11 blocked outputs kept as samples (5 Sonnet, 6 GPT), every one reports no device sharing, cites the requested-amount-to-income ratio in its claims, and still labels the pattern "unclear" where the rule would say affordability. Eleven samples is an observation, not a rate. The rule's thresholds were set in advance (C6) and the models are not necessarily wrong, but the downgrade guardrail means a quarter of cases would fall back to manual review in this configuration. Neither model beat the rule on pattern accuracy (about 46% against 58.8%). That is a workload and calibration question for the model owner, not evidence that the models are unsafe.

**C9 (Low, open). Prompt-injection effect is small and not yet measurable at this sample size.** See the reading note above. Neither model followed a note often enough to separate a real effect from noise, and "changed and not blocked" means the output moved, not that the move was harmful.

## Verdict

**The test bench is fit for purpose as a test bench, with conditions. No live assistant is validated.** Two live models have now been measured once (see "Live models"), which is a first measurement and not a validation. Conditions before any live result is relied on: (1) repeat the live runs with more cases per attack, several repeats and a second prompt, and report fault rates for the live models as well as attack rates; (2) add a calibrated semantic judge for C2; (3) bind claims to typed fact slots for C3; (4) stop passing untrusted notes to the generator for C5.

## What is not yet tested

- Repeatability. Each live model ran once, at temperature 0, with one prompt. There are no repeats, no second prompt and no confidence intervals.
- Seeded faults against a live model. The seeded-fault injectors run against stand-ins only.
- Semantic faithfulness, tone, and over- or under-confidence of free-text summaries.
- Latency, cost and rate-limit behaviour.
- Anything beyond synthetic data. The ground truth is known by construction, which makes the checks provable but also means real-world fraud will be messier.

## Method note

The test bench follows the "validate the validator" logic from Part B: each defect is injected into a known-good output, so a miss is a miss of the harness and not noise in the model. The evaluators are written as plain deterministic functions rather than with DeepEval, so they run with no network or API key; a model-graded evaluator is the planned addition for C2.
