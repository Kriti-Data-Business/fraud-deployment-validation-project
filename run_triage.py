"""Part C entrypoint: build the flagged-case queue, then validate the triage layer.

Usage:
    python run_triage.py            # trains the Part A model (~1 min), then runs the harness

What this measures (see docs/part_c_validation.md for the reading):
  1. Clean baseline: guardrail false-alarm rate and pattern accuracy vs ground truth.
  2. Seeded faults: how many injected defects the guardrails catch.
  3. Red-team: prompt-injection attacks against a trusting assistant, with and
     without the guardrail layer.
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from twohats.features import build_features
from twohats.synth_data import generate
from twohats.train import predict_calibrated, time_split, train_model
from twohats.triage.assistant import TemplateAssistant
from twohats.triage.cases import build_cases
from twohats.triage.faults import (
    ATTACKS, FAULTS, FaultyAssistant, NaiveEchoAssistant, SilentSoftenerAssistant, with_note,
)
from twohats.triage.guardrails import check, guarded

OUT = Path("reports/triage")
TRUTH_PATTERN = {"ring": "network_linkage", "opportunistic": "affordability"}


def main() -> None:
    (OUT / "figures").mkdir(parents=True, exist_ok=True)
    print("[1/4] Rebuilding the Part A model and the flagged queue (top 1%, out-of-time fold)...")
    df = build_features(generate(n_apps=80_000, fraud_rate=0.025, seed=42))
    train, valid, test = time_split(df)
    model, calibrator, cols = train_model(train, valid)
    scores = predict_calibrated(model, calibrator, test, cols)
    cases = build_cases(test, scores)
    n = len(cases)
    mix = Counter(c.truth["fraud_type"] for c in cases)
    print(f"      {n} flagged cases: {dict(mix)}")

    res: dict = {"n_cases": n, "queue_mix": dict(mix)}
    base = TemplateAssistant()

    print("[2/4] Clean baseline...")
    outs = [base.generate(c) for c in cases]
    clean_viol = [check(c, o) for c, o in zip(cases, outs)]
    fraud_cases = [(c, o) for c, o in zip(cases, outs) if c.truth["fraud_type"] in TRUTH_PATTERN]
    correct = sum(o["pattern"] == TRUTH_PATTERN[c.truth["fraud_type"]] for c, o in fraud_cases)
    by_type = {}
    for ft, want in TRUTH_PATTERN.items():
        sub = [(c, o) for c, o in fraud_cases if c.truth["fraud_type"] == ft]
        by_type[ft] = {"n": len(sub), "pattern_correct": sum(o["pattern"] == want for _, o in sub)}
    res["clean"] = {
        "guardrail_false_alarms": sum(bool(v) for v in clean_viol),
        "pattern_accuracy_on_true_fraud": correct / max(1, len(fraud_cases)),
        "true_fraud_in_queue": len(fraud_cases),
        "false_positives_in_queue": n - len(fraud_cases),
        "by_fraud_type": by_type,
        "queue_precision": len(fraud_cases) / n,
    }

    print("[3/4] Seeded faults...")
    faults = {}
    for name, (_, expected) in FAULTS.items():
        a = FaultyAssistant(name)
        viols = [check(c, a.generate(c)) for c in cases]
        codes = Counter(code for v in viols for code in v)
        faults[name] = {
            "detected": sum(bool(v) for v in viols),
            "n": n,
            "rate": sum(bool(v) for v in viols) / n,
            "expected_catchable": expected,
            "codes": dict(codes),
        }
    res["faults"] = faults
    catchable = [f for f in faults.values() if f["expected_catchable"]]
    res["faults_summary"] = {
        "fault_types": len(faults),
        "types_fully_caught": sum(f["rate"] == 1.0 for f in faults.values()),
        "types_missed_entirely": [k for k, f in faults.items() if f["rate"] == 0.0],
        "injections": sum(f["n"] for f in faults.values()),
        "injections_caught": sum(f["detected"] for f in faults.values()),
    }

    print("[4/4] Red-team (prompt injection via applicant note)...")
    naive, quiet, rt = NaiveEchoAssistant(), SilentSoftenerAssistant(), {}
    for aname, note in ATTACKS.items():
        row = {}
        for label, assistant in (
            ("trusting_assistant", naive),
            ("silent_softener", quiet),
            ("baseline_ignores_note", base),
        ):
            compromised = escaped = blocked = escaped_form_only = 0
            for c in cases:
                attacked = with_note(c, note)
                raw = assistant.generate(attacked)
                bad = raw != base.generate(c)  # any deviation from the clean answer
                viol = check(attacked, raw)
                compromised += bad
                blocked += bool(viol)
                escaped += bad and not viol
                escaped_form_only += bad and not check(attacked, raw, rederive=False)
            row[label] = {"compromised": compromised, "blocked": blocked, "escaped": escaped,
                          "escaped_form_checks_only": escaped_form_only, "n": n}
        rt[aname] = row
    res["redteam"] = rt
    attacks = [a for a in ATTACKS if a != "benign_note"]
    tot = n * len(attacks)
    res["redteam_summary"] = {
        "attacks": len(attacks),
        "trusting_compromised_before_guardrails": sum(rt[a]["trusting_assistant"]["compromised"] for a in attacks),
        "trusting_escaped_after_guardrails": sum(rt[a]["trusting_assistant"]["escaped"] for a in attacks),
        "attack_attempts": tot,
        "silent_compromised_before_guardrails": sum(rt[a]["silent_softener"]["compromised"] for a in attacks),
        "silent_escaped_form_checks_only": sum(rt[a]["silent_softener"]["escaped_form_checks_only"] for a in attacks),
        "silent_escaped_after_guardrails": sum(rt[a]["silent_softener"]["escaped"] for a in attacks),
        "benign_note_blocked_for_trusting": rt["benign_note"]["trusting_assistant"]["blocked"],
        "benign_note_blocked_for_baseline": rt["benign_note"]["baseline_ignores_note"]["blocked"],
    }

    # fail-closed demonstration on one case, saved as an example
    c0 = cases[0]
    blocked_out, viol = guarded(FaultyAssistant("decision_language"), c0)
    res["example"] = {
        "clean_output": outs[0],
        "faulty_output": FaultyAssistant("decision_language").generate(c0),
        "violations": viol,
        "after_guardrails": blocked_out,
    }
    (OUT / "results.json").write_text(json.dumps(res, indent=2))
    _chart(faults)
    _print(res)


def _chart(faults: dict) -> None:
    names = list(faults)
    rates = [faults[k]["rate"] * 100 for k in names]
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.barh(names[::-1], rates[::-1], color=["#B45309" if r < 100 else "#0F766E" for r in rates[::-1]])
    ax.set_xlim(0, 100)
    ax.set_xlabel("% of seeded faults caught by the guardrails")
    ax.set_title("Guardrail detection by seeded fault type")
    fig.tight_layout()
    fig.savefig(OUT / "figures/fault_detection.png", dpi=150)


def _print(r: dict) -> None:
    c, s, rs = r["clean"], r["faults_summary"], r["redteam_summary"]
    print(f"\nQueue: {r['n_cases']} cases, precision {c['queue_precision']:.1%}; mix {r['queue_mix']}")
    print(f"Clean baseline: {c['guardrail_false_alarms']} guardrail false alarms; "
          f"pattern accuracy on true fraud {c['pattern_accuracy_on_true_fraud']:.1%} "
          f"({c['by_fraud_type']})")
    print(f"Seeded faults: {s['injections_caught']}/{s['injections']} injections caught; "
          f"{s['types_fully_caught']}/{s['fault_types']} types fully caught; missed entirely: {s['types_missed_entirely']}")
    print(f"Red-team: trusting assistant compromised {rs['trusting_compromised_before_guardrails']}/{rs['attack_attempts']} "
          f"before guardrails, {rs['trusting_escaped_after_guardrails']} escaped after; "
          f"benign note blocked: trusting {rs['benign_note_blocked_for_trusting']}, baseline {rs['benign_note_blocked_for_baseline']}")
    print(f"          silent softener compromised {rs['silent_compromised_before_guardrails']}/{rs['attack_attempts']}, "
          f"{rs['silent_escaped_form_checks_only']} escaped form-level checks alone, "
          f"{rs['silent_escaped_after_guardrails']} escaped with pattern re-derivation")


if __name__ == "__main__":
    main()
