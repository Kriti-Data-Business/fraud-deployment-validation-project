"""Embed demo/demo_data.json into demo/template.html -> demo/index.html (self-contained)."""
import json
from pathlib import Path

root = Path(__file__).parent.parent / "demo"
data = json.dumps(json.loads((root / "demo_data.json").read_text()), separators=(",", ":")).replace("</", "<\\/")
html = (root / "template.html").read_text().replace("/*__DATA__*/", data)
(root / "index.html").write_text(html)
print(f"demo/index.html {len(html)/1024:.0f} KB")
