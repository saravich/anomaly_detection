import argparse, yaml, os
from glob import glob
from ..classical import build_template, lbp_iforest_fit, save_model
from ..data import make_synth

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", required=True, help="Output model .pkl")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    root = cfg["dataset"]["root"]
    size = tuple(cfg["dataset"]["image_size"])
    method = cfg["method"]
    sub = cfg.get("classical", {}).get("submethod", "ssim_template")

    if cfg["dataset"]["name"] == "synth":
        make_synth(root)

    train_glob = os.path.join(root, "train", "good", "*.png")
    train_paths = sorted(glob(train_glob))
    assert train_paths, f"No training images at {train_glob}"

    if method == "classical" and sub == "ssim_template":
        ref = build_template(train_paths, size)
        model = {"type": "ssim_template", "size": size, "ref": ref}
    elif method == "classical" and sub == "lbp_iforest":
        clf = lbp_iforest_fit(train_paths, size)
        model = {"type": "lbp_iforest", "size": size, "clf": clf}
    else:
        raise ValueError(f"Unknown method: {method} / {sub}")

    save_model(model, args.out)
    print("Wrote model:", args.out)

if __name__ == "__main__":
    main()
