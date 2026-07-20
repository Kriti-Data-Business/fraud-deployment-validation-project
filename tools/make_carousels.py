"""Generate the two LinkedIn carousel PDFs (4:5 portrait, one slide per page).

Carousel 1 — the Two Hats story: built a fraud model, then validated it
             like a bank would; what the validation found.
Carousel 2 — the field guide: 10 silent failure modes of fraud models and
             the checks that catch them.

Usage: python tools/make_carousels.py   (outputs to linkedin/)
"""

from __future__ import annotations

from pathlib import Path

from reportlab.lib.colors import HexColor
from reportlab.lib.utils import simpleSplit
from reportlab.pdfgen import canvas

W, H = 540, 675  # 4:5 portrait, LinkedIn's tallest carousel format
M = 52           # margin

OUT = Path(__file__).resolve().parent.parent / "linkedin"


class Deck:
    def __init__(self, path: Path, bg: str, fg: str, accent: str, muted: str):
        self.c = canvas.Canvas(str(path), pagesize=(W, H))
        self.bg, self.fg, self.accent, self.muted = map(HexColor, (bg, fg, accent, muted))
        self.n = 0
        self.total = 0  # set before save via two-pass? simpler: set manually

    # -- primitives ---------------------------------------------------------
    def start(self):
        self.n += 1
        self.c.setFillColor(self.bg)
        self.c.rect(0, 0, W, H, stroke=0, fill=1)

    def end(self, footer=True):
        if footer:
            self.c.setFont("Helvetica", 10.5)
            self.c.setFillColor(self.muted)
            self.c.drawString(M, 30, "Kriti Yadav · Data Science & AI")
            self.c.drawRightString(W - M, 30, f"{self.n:02d}")
            self.c.setFillColor(self.accent)
            self.c.rect(M, 46, W - 2 * M, 1.2, stroke=0, fill=1)
        self.c.showPage()

    def kicker(self, text, y, color=None):
        self.c.setFont("Helvetica-Bold", 12.5)
        self.c.setFillColor(color or self.accent)
        self.c.drawString(M, y, text.upper())
        return y - 26

    def head(self, text, y, size=33, color=None, leading=None, maxw=None):
        self.c.setFont("Helvetica-Bold", size)
        self.c.setFillColor(color or self.fg)
        leading = leading or size * 1.18
        for line in simpleSplit(text, "Helvetica-Bold", size, maxw or (W - 2 * M)):
            self.c.drawString(M, y, line)
            y -= leading
        return y

    def body(self, text, y, size=15.5, color=None, maxw=None, leading=None, x=None):
        self.c.setFont("Helvetica", size)
        self.c.setFillColor(color or self.fg)
        leading = leading or size * 1.42
        for line in simpleSplit(text, "Helvetica", size, maxw or (W - 2 * M)):
            self.c.drawString(x or M, y, line)
            y -= leading
        return y

    def stat(self, value, label, y, size=58, color=None):
        self.c.setFont("Helvetica-Bold", size)
        self.c.setFillColor(color or self.accent)
        self.c.drawString(M, y, value)
        y -= 24
        self.c.setFont("Helvetica", 14.5)
        self.c.setFillColor(self.fg)
        for line in simpleSplit(label, "Helvetica", 14.5, W - 2 * M):
            self.c.drawString(M, y, line)
            y -= 20
        return y

    def chip(self, text, y, color=None):
        color = color or self.accent
        self.c.setFillColor(color)
        w = self.c.stringWidth(text.upper(), "Helvetica-Bold", 11.5) + 20
        self.c.roundRect(M, y - 6, w, 24, 12, stroke=0, fill=1)
        self.c.setFillColor(self.bg)
        self.c.setFont("Helvetica-Bold", 11.5)
        self.c.drawString(M + 10, y, text.upper())
        return y - 34

    def rule(self, y):
        self.c.setFillColor(self.muted)
        self.c.rect(M, y, W - 2 * M, 0.8, stroke=0, fill=1)
        return y - 18

    def save(self):
        self.c.save()


# ---------------------------------------------------------------------------
# Carousel 1 — the Two Hats story
# ---------------------------------------------------------------------------
def carousel_story():
    NAVY, INK, BLUE, MUT = "#0B1F3A", "#F5F7FA", "#4DA3FF", "#7A8CA6"
    AMBER, RED, GREEN = "#FFB020", "#FF6B6B", "#3DDC97"
    d = Deck(OUT / "carousel_1_two_hats_story.pdf", NAVY, INK, BLUE, MUT)

    # 1 — hook
    d.start()
    y = d.kicker("A fraud-model post-mortem — before deployment", H - 92)
    y = d.head("My fraud model hit PR-AUC 0.87.", y - 16, 36)
    y = d.head("The validation report still filed 8 findings against it.", y - 6, 36, color=HexColor(AMBER))
    y = d.body("I built the model. Then I switched roles and challenged it the way a bank's Model Risk team would.", y - 22, 17)
    d.body("Swipe for what the metrics were hiding →", 96, 15, color=HexColor(BLUE))
    d.end()

    # 2 — why
    d.start()
    y = d.kicker("Why bother?", H - 92)
    y = d.head("Banks run three lines of defence. Portfolios only ever show Line 1.", y - 14, 30)
    y = d.body("Line 1 builds the model. Line 2 — Model Risk — independently proves whether it's fit for purpose. Regulators have required this since 2011 (SR 11-7), and still do under its 2026 replacement, SR 26-2.", y - 18)
    y = d.body("So this project wears both hats, honestly: same model, same data, a wall between the roles.", y - 10)
    d.end()

    # 3 — the build
    d.start()
    y = d.kicker("Hat 1 — the developer", H - 92)
    y = d.head("A production-shaped application-fraud model", y - 14, 28)
    y = d.body("80k applications · fraud rings sharing devices & addresses · point-in-time features · time-ordered split · calibrated LightGBM.", y - 16, 15)
    y = d.stat("0.873", "PR-AUC on the out-of-time test fold", y - 46)
    y = d.stat("99.2%", "precision in the top-1% investigation queue", y - 40)
    y = d.body("Looks shippable. Most projects stop here.", y - 26, 16, color=HexColor(AMBER))
    d.end()

    # 4 — hat switch
    d.start()
    y = d.kicker("Hat 2 — the validator", H - 92)
    y = d.head("Then I tried to break my own model.", y - 14, 32)
    y = d.body("The independent validation suite:", y - 16, 16)
    for item in [
        "Challenger models — must the champion be this complex?",
        "Feature-family ablation — any single point of failure?",
        "Input perturbation — can an applicant steer the score?",
        "Label-quality scan (confident learning)",
        "PSI drift analysis across time windows",
        "SHAP review — would explanations survive challenge?",
        "Segment analysis — who does it miss?",
    ]:
        d.c.setFillColor(HexColor(BLUE)); d.c.circle(M + 4, y + 5, 2.6, stroke=0, fill=1)
        y = d.body(item, y, 14.5, x=M + 16, maxw=W - 2 * M - 16) - 4
    d.end()

    # 5 — finding 1
    d.start()
    y = d.chip("Finding 1 · High", H - 88, HexColor(RED))
    y = d.head("A simpler model beat the champion.", y - 14, 32)
    y = d.stat("0.887 vs 0.873", "challenger PR-AUC vs champion — a 300-tree shallow GBM outperformed the tuned 2000-tree model", y - 52, 44)
    y = d.body("The champion's complexity was unearned. Nobody checks this unless a challenger benchmark forces the question.", y - 26)
    d.end()

    # 6 — finding 2
    d.start()
    y = d.chip("Finding 2 · High", H - 88, HexColor(RED))
    y = d.head('My "multi-factor" model was secretly a one-trick model.', y - 14, 30)
    y = d.body("Remove all 11 entity-linkage features: PR-AUC barely moves (−0.004).", y - 16, 15.5)
    y = d.stat("0.873 → 0.466", "PR-AUC when the 4 affordability ratios are removed", y - 48, 40)
    y = d.body("The stated design theory and the model's actual behaviour were two different things. Ablation is how you find out.", y - 26)
    d.end()

    # 7 — finding 3
    d.start()
    y = d.chip("Finding 3 · High", H - 88, HexColor(RED))
    y = d.head("An applicant could steer the queue.", y - 14, 32)
    y = d.stat("92%", "of the top-1% investigation queue turns over if applicants understate income by 20%", y - 52)
    y = d.body("The dominant feature is self-reported and unverified. That's not a model bug — it's an attack surface. Condition of approval: income verification for high-score applications.", y - 26)
    d.end()

    # 8 — more findings
    d.start()
    y = d.kicker("Findings 4–8", H - 92)
    y = d.head("And the quieter ones:", y - 14, 30)
    for t, s in [
        ("Segment blind spot — 90% of ring fraud caught, only 57% of solo fraud. Headline metrics averaged it away.", AMBER),
        ("Drift-alarm fatigue — 9 of 24 features breach PSI thresholds while the score is perfectly stable. Naive monitoring would cry wolf daily.", AMBER),
        ("Age-proxy risk — age carries high attribution; fine for triage, a discrimination problem the day scores gate decisions.", AMBER),
    ]:
        d.c.setFillColor(HexColor(s)); d.c.rect(M, y + 2, 3, 13, stroke=0, fill=1)
        y = d.body(t, y, 14.5, x=M + 14, maxw=W - 2 * M - 14) - 10
    d.end()

    # 9 — verdict
    d.start()
    y = d.kicker("The verdict", H - 92)
    y = d.head("Fit for purpose — with conditions.", y - 16, 34, color=HexColor(GREEN))
    y = d.body("Approved for human-reviewed triage only, conditional on fixing the three High findings and a monitoring plan with governance-calibrated thresholds.", y - 18)
    y = d.body("Same model, different intended use — auto-decline instead of triage — and the verdict flips to NOT FIT. Purpose is part of the model.", y - 12, 15.5, color=HexColor(AMBER))
    d.end()

    # 10 — CTA
    d.start()
    y = d.kicker("The point", H - 92)
    y = d.head("7 of these 8 findings are invisible to train → test → ship.", y - 14, 30)
    y = d.body("They only surface when someone whose job is NOT shipping the model goes looking. That's what AI validation is — and why it's a discipline, not a checkbox.", y - 16)
    y = d.body("Everything here is reproducible: synthetic data with known ground truth, single-command pipeline, full validation report on my GitHub (Kriti-Data-Business).", y - 12)
    d.body("Next: validating an LLM fraud-triage assistant — the part current regulation doesn't even cover yet. Follow along.", y - 14, 15.5, color=HexColor(BLUE))
    d.end()

    d.save()


# ---------------------------------------------------------------------------
# Carousel 2 — the field guide
# ---------------------------------------------------------------------------
def carousel_guide():
    INKBG, INK, GREEN, MUT = "#101418", "#F2F4F6", "#3DDC97", "#71808F"
    AMBER, RED = "#FFB020", "#FF6B6B"
    d = Deck(OUT / "carousel_2_silent_failure_modes.pdf", INKBG, INK, GREEN, MUT)

    scenarios = [
        ("Leakage by resampling", "SMOTE or scaling applied before the train/test split; random splits leak future fraud-ring members into training.", "Audit pipeline order. Time-ordered splits. Point-in-time features only.", False),
        ("The delayed-label blind period", "Fraud labels arrive 1–6 months late. The model decays silently while dashboards show stale wins.", "Annotate label lag on every metric. Use early proxy signals (disputes, soft labels).", False),
        ("Selective-labels feedback loop", "Only investigated cases get labels — the model retrains on its own past decisions and calcifies its blind spots.", "Sample and label below the threshold. Validator-side label audits.", False),
        ("Adversarial adaptation", "Fraud is a cat-and-mouse game. Fraudsters probe thresholds and route around static models.", "Perturbation-test manipulable inputs. (Mine: −20% stated income flipped 92% of the queue.)", True),
        ("Hidden single point of failure", "One feature family quietly carries the model. An outage or evasion of it collapses detection.", "Family ablation. (Mine: PR-AUC 0.87 → 0.47 without affordability ratios.)", True),
        ("Unearned complexity", "The tuned champion ships because it's the champion. Nobody asks if simpler matches it.", "Challenger benchmark. (Mine: the challenger won, 0.887 vs 0.873.)", True),
        ("Segment blind spots", "Aggregate metrics look great while a channel or fraud type is under-detected.", "Slice everything: fraud type, channel, amount band. (Mine: 90% ring vs 57% solo capture.)", True),
        ("Drift-alarm fatigue", "Uniform PSI thresholds on bursty features alarm daily — until the team ignores all alerts.", "Calibrate thresholds per feature family; monitor score PSI first. (Mine: 9 features 'drifting', score stable.)", True),
        ("Proxy discrimination", "Age or postcode enter as innocent features — and become indirect discrimination the day scores gate decisions.", "Attribution review on protected attributes + a fairness gate before scope expands. (Mine: age ranked top-6 by SHAP.)", True),
        ("Scores that aren't probabilities", "Rank-good, probability-bad scores silently corrupt expected-loss maths and risk thresholds.", "Reliability curves and Brier on the out-of-time fold — not the training set.", False),
    ]

    # 1 — hook
    d.start()
    y = d.kicker("A field guide for ML in fraud & credit", H - 92)
    y = d.head("A fraud model can look perfect and still be broken.", y - 16, 34)
    y = d.head("10 silent failure modes — and the checks that catch them.", y - 8, 26, color=HexColor(GREEN))
    y = d.body("I validated my own fraud model the way a bank's Model Risk team would. Six of these ten failures were live in it — the red-tagged slides are the ones I caught in my own build.", y - 20, 16)
    d.body("Save this for your next model review →", 96, 15, color=HexColor(GREEN))
    d.end()

    # 2..11 — scenarios
    for i, (title, how, check, caught) in enumerate(scenarios, 1):
        d.start()
        y = d.chip(f"Failure mode {i:02d}" + ("  ·  caught in my own model" if caught else ""), H - 88,
                   HexColor(RED) if caught else HexColor("#3A75C4"))
        y = d.head(title, y - 14, 30)
        y = d.body(how, y - 14, 16)
        y -= 10
        d.c.setFillColor(HexColor(GREEN))
        d.c.setFont("Helvetica-Bold", 13)
        d.c.drawString(M, y, "THE CHECK")
        y -= 22
        y = d.body(check, y, 15.5, color=HexColor(GREEN))
        d.end()

    # 12 — close
    d.start()
    y = d.kicker("The meta-lesson", H - 92)
    y = d.head("7 of 10 are invisible to the standard workflow: train → test → report AUC → ship.", y - 14, 29)
    y = d.body("They surface only under independent validation — which is why regulators have required it since SR 11-7 (2011), and why AI validation is one of the most underrated career paths in data science right now.", y - 18)
    y = d.body("Full validation report, findings log and reproducible code: github.com/Kriti-Data-Business", y - 12)
    d.body("Part 1 of this series: the Two Hats story. Part 3: validating an LLM assistant. Follow for the rest.", y - 14, 15, color=HexColor(GREEN))
    d.end()

    d.save()


if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    carousel_story()
    carousel_guide()
    print(f"Wrote {OUT / 'carousel_1_two_hats_story.pdf'}")
    print(f"Wrote {OUT / 'carousel_2_silent_failure_modes.pdf'}")
