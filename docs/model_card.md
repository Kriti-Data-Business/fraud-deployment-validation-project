# Model Card — Application Fraud Triage Model v1

| | |
|---|---|
| **Model** | LightGBM binary classifier + isotonic calibration ("champion") |
| **Purpose** | Prioritise incoming credit applications for fraud investigation (top-k% review queue) |
| **Intended use** | Triage/prioritisation only — ranks applications for human investigators |
| **Explicit non-uses** | Not a credit-worthiness score; not an auto-decline mechanism; scores must not gate approval without human review |
| **Population** | All incoming applications, all channels (online/branch/broker/mobile) |
| **Training data** | Synthetic application stream, 80k apps / 180 days, 2.5% fraud (ring + opportunistic patterns), ground truth by construction |
| **Features** | Point-in-time only: entity velocity (7/30d), expanding entity degree, affordability ratios, behavioural |
| **Split** | Time-ordered 70/15/15 (train/valid/test); test is strictly out-of-time |
| **Owner (Hat 1)** | Developer role — Part A of this repo |
| **Validator (Hat 2)** | Independent role — Part B, see [validation_report.md](validation_report.md) |

## Assumptions register (developer-stated, validator-challenged)

1. Fraud patterns are sufficiently stationary over a 6-month window for a static model.
2. Entity resolution (device/address/phone matching) is reliable upstream.
3. Fraud labels are complete and timely (known false in production — delayed labels; see validation report F7).
4. Stated income is informative despite being self-reported (challenged — see F3).
5. Both fraud types (ring, opportunistic) are in scope for a single model (challenged — see F2/F4).
