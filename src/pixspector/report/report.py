from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Any, Optional

from jinja2 import Environment, FileSystemLoader, select_autoescape
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import cm


def save_json_report(data: Dict[str, Any], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(data, f, indent=2)


def save_pdf_report(
    data: Dict[str, Any],
    out_path: Path,
    visuals_dir: Optional[Path] = None,
) -> None:
    """
    Generate a simple PDF forensic report.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)

    doc = SimpleDocTemplate(str(out_path), pagesize=A4)
    styles = getSampleStyleSheet()
    flow = []

    flow.append(Paragraph("<b>pixspector Forensic Report</b>", styles["Title"]))
    flow.append(Spacer(1, 0.5 * cm))

    flow.append(Paragraph(f"<b>Input file:</b> {data.get('input')}", styles["Normal"]))
    flow.append(Paragraph(f"<b>SHA-256:</b> {data.get('sha256', 'n/a')}", styles["Normal"]))
    flow.append(Paragraph(f"<b>Suspicion Index:</b> {data.get('suspicion_index', 'n/a')} "
                          f"({data.get('bucket_label', '')})", styles["Normal"]))
    flow.append(Spacer(1, 0.5 * cm))

    decision = data.get("decision", {})
    decision_level = decision.get("confidence_level")
    decision_rationale = decision.get("rationale") or []
    if decision_level:
        flow.append(Paragraph("<b>Decision</b>", styles["Heading2"]))
        flow.append(Paragraph(f"<b>Confidence Level:</b> {decision_level}", styles["Normal"]))
        if decision_rationale:
            for item in decision_rationale:
                flow.append(Paragraph(f"- {item}", styles["Normal"]))
        flow.append(Spacer(1, 0.5 * cm))

    flow.append(Paragraph("<b>Evidence Items</b>", styles["Heading2"]))
    ev = data.get("evidence", [])
    if not ev:
        flow.append(Paragraph("No strong forensic cues found.", styles["Normal"]))
    else:
        for e in ev:
            text = f"- {e['key']} (weight {e['weight']}): {e['rationale']}"
            if e.get("value") is not None:
                text += f" (value={e['value']})"
            flow.append(Paragraph(text, styles["Normal"]))
    flow.append(Spacer(1, 0.5 * cm))

    flow.append(Paragraph("<b>Notes</b>", styles["Heading2"]))
    for n in data.get("notes", []):
        flow.append(Paragraph(f"- {n}", styles["Normal"]))

    # Optionally append visuals
    if visuals_dir and visuals_dir.exists():
        flow.append(Spacer(1, 0.5 * cm))
        flow.append(Paragraph("<b>Visual Evidence</b>", styles["Heading2"]))
        for img_path in sorted(visuals_dir.glob("*.png")):
            flow.append(Paragraph(img_path.name, styles["Normal"]))
            flow.append(Image(str(img_path), width=12 * cm, height=8 * cm))
            flow.append(Spacer(1, 0.3 * cm))

    doc.build(flow)
