import argparse, json, numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score
from ..utils import load_json, save_json

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds", required=True, help="Folder containing preds.json")
    ap.add_argument("--out", required=True, help="Output metrics.json")
    args = ap.parse_args()

    preds = load_json(f"{args.preds}/preds.json")
    y = np.array([p["label"] for p in preds], dtype=int)
    s = np.array([p["score"] for p in preds], dtype=float)
    metrics = {}
    try:
        metrics["auroc"] = float(roc_auc_score(y, s))
        metrics["auprc"] = float(average_precision_score(y, s))
    except Exception as e:
        metrics["auroc"] = None
        metrics["auprc"] = None
        metrics["error"] = str(e)
    thr = float(np.percentile(s, 95))
    yhat = (s >= thr).astype(int)
    metrics["threshold"] = thr
    metrics["acc@p95"] = float((yhat == y).mean())

    save_json(metrics, args.out)
    print("Wrote metrics:", args.out)

if __name__ == "__main__":
    main()
