import argparse, json, os, glob
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--metrics", required=True)
    ap.add_argument("--preds", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    styles = getSampleStyleSheet()
    doc = SimpleDocTemplate(args.out, pagesize=A4)
    story = []
    story.append(Paragraph("Anomaly Detection Report", styles["Title"]))
    story.append(Spacer(1, 8))

    with open(args.metrics, "r") as f:
        m = json.load(f)
    rows = [["Metric", "Value"]]
    for k, v in m.items():
        if isinstance(v, float):
            rows.append([k, f"{v:.4f}"])
        else:
            rows.append([k, str(v)])
    tbl = Table(rows, colWidths=[220, 200])
    tbl.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,0), colors.lightgrey),
        ("GRID", (0,0), (-1,-1), 0.25, colors.grey),
        ("ALIGN", (1,1), (-1,-1), "RIGHT"),
    ]))
    story.append(tbl)
    story.append(Spacer(1, 12))

    story.append(Paragraph("Qualitative Examples", styles["Heading2"]))
    story.append(Spacer(1, 6))

    heatmaps = sorted(glob.glob(os.path.join(args.preds, "*_heat.png")))[:6]
    if not heatmaps:
        story.append(Paragraph("No heatmaps found (LBP-only run or missing _heat.png files).", styles["Normal"]))
    else:
        for hp in heatmaps:
            story.append(Image(hp, width=380, height=285))
            story.append(Spacer(1, 6))

    doc.build(story)
    print("Wrote report:", args.out)

if __name__ == "__main__":
    main()
