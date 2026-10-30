"""Export real harness results for the interactive Part C demo.

The demo never re-implements the guardrails. This script runs the actual Python
harness over a stratified sample of the flagged queue and stores every output
and every guardrail verdict, so the page only plays back what the code produced.

Usage:  python tools/export_demo_data.py [out.json]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from twohats.features import build_features
from twohats.synth_data import generate
from twohats.train import predict_calibrated, time_split, train_model
from twohats.triage.assistant import TemplateAssistant, derive_pattern
from twohats.triage.cases import build_cases
from twohats.triage.faults import (
    ATTACKS, FAULTS, FaultyAssistant, NaiveEchoAssistant, SilentSoftenerAssistant, with_note,
)
from twohats.triage.guardrails import check

OUT = Path(sys.argv[1] if len(sys.argv) > 1 else "demo/demo_data.json")


def pick(cases):
    """Stratified, deterministic sample: every unclear and false-positive case, plus spread of the rest."""
    chosen = {}
    for i, c in enumerate(cases):
        if derive_pattern(c) == "unclear" or c.truth["fraud_type"] == "none":
            chosen[i] = c
    for pattern, k in (("network_linkage", 12), ("affordability", 9)):
        idx = [i for i, c in enumerate(cases) if derive_pattern(c) == pattern and i not in chosen]
        step = max(1, len(idx) // k)
        for i in idx[::step][:k]:
            chosen[i] = cases[i]
    return [chosen[i] for i in sorted(chosen)]


def verdict(case, out):
    v = check(case, out)
    return {
        "output": out,
        "violations": v,
        "violations_form_only": check(case, out, rederive=False),
        "delivered": not v,
    }


def main() -> None:
    df = build_features(generate(n_apps=80_000, fraud_rate=0.025, seed=42))
    train, valid, test = time_split(df)
    model, calibrator, cols = train_model(train, valid)
    cases = build_cases(test, predict_calibrated(model, calibrator, test, cols))
    sample = pick(cases)
    base, naive, quiet = TemplateAssistant(), NaiveEchoAssistant(), SilentSoftenerAssistant()

    out_cases = []
    for c in sample:
        modes = {"clean": verdict(c, base.generate(c))}
        for name in FAULTS:
            modes[f"fault:{name}"] = verdict(c, FaultyAssistant(name).generate(c))
        for aname, note in ATTACKS.items():
            attacked = with_note(c, note)
            modes[f"attack:{aname}:trusting"] = {**verdict(attacked, naive.generate(attacked)), "note": note}
            modes[f"attack:{aname}:quiet"] = {**verdict(attacked, quiet.generate(attacked)), "note": note}
        out_cases.append(
            {
                "id": c.case_id,
                "score": round(c.score, 4),
                "facts": [{"id": f.id, "name": f.name, "value": f.value} for f in c.facts.values()],
                "truth": c.truth["fraud_type"],
                "derived_pattern": derive_pattern(c),
                "modes": modes,
            }
        )

    summary = json.loads(Path("reports/triage/results.json").read_text())
    payload = {
        "meta": {
            "queue_size": summary["n_cases"],
            "sample_size": len(out_cases),
            "note": "Precomputed by run_triage.py components. Stand-in assistants, synthetic data, no live LLM.",
        },
        "attacks": ATTACKS,
        "faults": {k: bool(v[1]) for k, v in FAULTS.items()},
        "summary": {
            "faults": summary["faults_summary"],
            "redteam": summary["redteam_summary"],
            "clean": summary["clean"],
        },
        "cases": out_cases,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, separators=(",", ":")))
    print(f"{len(out_cases)} cases x {len(out_cases[0]['modes'])} modes -> {OUT} ({OUT.stat().st_size/1024:.0f} KB)")


if __name__ == "__main__":
    main()
