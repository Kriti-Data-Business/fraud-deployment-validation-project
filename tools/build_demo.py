"""Embed demo/demo_data.json into demo/template.html -> demo/index.html (self-contained)."""
import json
from pathlib import Path

root = Path(__file__).parent.parent / "demo"
payload = json.loads((root / "demo_data.json").read_text())
# Real-model results, if the live run has been done: reports/triage/live_*.json
payload["live"] = [json.loads(p.read_text()) for p in sorted((root.parent / "reports/triage").glob("live_*.json"))]
data = json.dumps(payload, separators=(",", ":")).replace("</", "<\\/")
html = (root / "template.html").read_text().replace("/*__DATA__*/", data)
(root / "index.html").write_text(html)
print(f"demo/index.html {len(html)/1024:.0f} KB; live models embedded: {len(payload['live'])}")
