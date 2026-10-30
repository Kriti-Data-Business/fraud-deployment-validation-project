# Part C: validating the LLM fraud-triage layer

**Status:** the test bench is built and run. A live LLM has **not** yet been run through it. Every result below comes from deterministic stand-in assistants, so it measures the harness and the guardrails, not the behaviour of a real model. The live run is the next step (see "What is not yet tested").

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

## Findings

**C1 (High, partly closed). Quiet compliance with injected instructions passes every form-level check.** The quiet assistant keeps the output well-formed, keeps numbers grounded and leaves the risk band alone, but downgrades the pattern to "unclear" and the next step to manual review. All 720 compromised outputs passed. Adding an independent re-derivation of the pattern from the facts, applied one way only (a summary may not lower scrutiny the facts support), cut escapes to 12. Those 12 are 2 cases across 6 attacks where the facts only support "unclear" and the attacker dropped the evidence line, so the remaining gap is evidence omission. I wrote this attacker to be invisible to the form checks, so this shows that the failure class exists and that the control works against it, not how common it is.

**C2 (High, open). A false claim with no numbers cannot be detected.** "The applicant previously defaulted with another lender" passed 120 of 120 times. The deterministic checks verify form, numbers and wording, not truth. This needs a semantic judge, calibrated against human labels, before a live model is trusted.

**C3 (Medium, open). Number grounding is weaker than it looks.** The fabricated-number fault was flagged in all 120 cases, but number grounding caught only 113. The other 7 were flagged because the fake sentence contained the word "declined", which is a coincidence. An invented number that equals a cited fact's value passes. Typed claim templates, where each claim slot is bound to one fact, would close this.

**C4 (Medium, open). The assistant inherits the model's blind spot.** Even the baseline's pattern labels are right far less often for opportunistic fraud (39%) than for rings (75%). That echoes Part B finding F4 (ring capture 90.0%, opportunistic 57.5%). The two measure different things, so this is a pointer, not a proof. An "unclear" label must never be read as evidence of absence.

**C5 (Medium, open). Fail-closed over-blocks.** The trusting assistant's benign note was blocked 120 of 120 times. That is the right outcome for a system that should never repeat notes, but it means notes should not reach the generator at all, or should pass through a separate extraction step.

**C6 (Low, known). Rule thresholds were set a priori.** The pattern rule (3 shared entities, 0.5 amount-to-income) was not tuned on the test fold, and tuning it there would be leakage. It would need a validation-fold calibration before any use beyond a baseline.

## Verdict

**The test bench is fit for purpose as a test bench, with conditions. The live assistant is not yet validated.** Conditions before any live-model result is reported: (1) run the live model through the same harness and report its fault and attack rates separately; (2) add a calibrated semantic judge for C2; (3) bind claims to typed fact slots for C3; (4) stop passing untrusted notes to the generator for C5.

## What is not yet tested

- Any real LLM. The adapter exists and fails closed on non-JSON output, but it has only been exercised with a fake client.
- Semantic faithfulness, tone, and over- or under-confidence of free-text summaries.
- Latency, cost and rate-limit behaviour.
- Anything beyond synthetic data. The ground truth is known by construction, which makes the checks provable but also means real-world fraud will be messier.

## Method note

The test bench follows the "validate the validator" logic from Part B: each defect is injected into a known-good output, so a miss is a miss of the harness and not noise in the model. The evaluators are written as plain deterministic functions rather than with DeepEval, so they run with no network or API key; a model-graded evaluator is the planned addition for C2.
