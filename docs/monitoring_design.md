# Continuous monitoring and fallback design for the triage assistant

**Status:** design only. Nothing here is implemented as running monitoring. Thresholds are proposed starting values to be agreed with the model owner and first-line operations, not tuned numbers.

## Principle

The assistant prepares; people decide. Every control below exists so that when the assistant is wrong, the cost is "an investigator reads the raw facts", never "a wrong decision".

## What to log for every case

Case id, model and prompt version, input facts, the raw assistant output, every guardrail code fired, whether the fallback was used, investigator action, and whether the investigator edited or overrode the summary. No free-text applicant note is logged in clear form.

## Signals and proposed starting thresholds

| Signal | Why it matters | Proposed trigger |
|---|---|---|
| Fallback rate (any guardrail fired) | Rising rate means the model or the inputs have changed | Review if above 5% over a day; pause if above 15% |
| Guardrail mix | A single code surging (for example `ungrounded_number`) points at one failure mode | Review on a 3x jump against the 30-day baseline |
| Investigator override or edit rate | The only signal that sees faults the guardrails cannot | Review if above 20% on a weekly sample |
| Score and queue drift (PSI on the score, pattern mix in the queue) | The assistant's inputs have shifted | Reuse the Part B PSI approach; trigger on a sustained breach, not a single window |
| Golden-set regression | Catches silent model or prompt changes | Run on every model, prompt or guardrail change; block release if any previously caught fault type is missed |
| Seeded-fault canaries | Proves the guardrails are still alive | Inject a known fault into a small share of traffic in staging weekly; alert if not caught |

## Fallback and kill switch

1. **Per case:** any guardrail violation replaces the AI summary with the manual-review fallback. The investigator sees the facts and the model score only.
2. **Per segment:** if a trigger above fires for one fraud pattern, switch off summaries for that pattern only.
3. **Global:** a single switch turns off the assistant. The queue and the Part A model are unaffected, so the process degrades to its pre-assistant state.

## Change control

A change to the model, the prompt, the facts schema or the guardrails triggers a re-run of the full harness (`python run_triage.py` plus the live-model run) and a short written comparison against the last validated version.

## Known gaps this design does not close

- Semantic truth of text without numbers (finding C2). Needs a calibrated judge and the override-rate signal as a backstop.
- Evidence omission (finding C1 residual). Needs a minimum-evidence rule per pattern.
- Fairness of summaries across groups. Age is kept out of the facts, but proxies in other facts have not been tested for this layer.
