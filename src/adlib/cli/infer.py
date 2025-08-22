import argparse, yaml, os
from glob import glob
import numpy as np, cv2 as cv
from ..classical import load_model, ssim_map, lbp_iforest_score
from ..utils import imread_gray, resize, ensure_dir, save_json
from ..viz import overlay_heatmap

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    size = tuple(cfg["dataset"]["image_size"])

    model = load_model(args.model)
    ensure_dir(args.out)

    root = cfg["dataset"]["root"]
    good = sorted(glob(os.path.join(root, "test", "good", "*.png")))
    anom = sorted(glob(os.path.join(root, "test", "anomaly", "*.png")))
    labeled = [(p, 0) for p in good] + [(p, 1) for p in anom]

    preds = []
    for path, label in labeled:
        im = resize(imread_gray(path), size)
        if model["type"] == "ssim_template":
            heat, score = ssim_map(im, model["ref"])
            overlay = overlay_heatmap(im, heat)
            cv.imwrite(os.path.join(args.out, os.path.basename(path).replace(".png", "_heat.png")), overlay)
        elif model["type"] == "lbp_iforest":
            score = lbp_iforest_score(model["clf"], im)
        else:
            raise ValueError("Unknown model type")

        preds.append({"path": path, "label": label, "score": float(score)})
    save_json(preds, os.path.join(args.out, "preds.json"))
    print("Wrote predictions:", os.path.join(args.out, "preds.json"))

if __name__ == "__main__":
    main()
