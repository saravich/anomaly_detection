import os, argparse, numpy as np
from PIL import Image, ImageDraw
from .utils import ensure_dir

def _noise_texture(h, w, seed=0):
    rng = np.random.default_rng(seed)
    base = rng.normal(127, 40, size=(h, w)).clip(0,255).astype(np.uint8)
    for _ in range(2):
        base = (base + np.roll(base, 1, axis=0) + np.roll(base, -1, axis=0) + 
                np.roll(base, 1, axis=1) + np.roll(base, -1, axis=1)) // 5
    return base

def _inject_defect(im, seed=0):
    rng = np.random.default_rng(seed)
    pil = Image.fromarray(im.copy())
    draw = ImageDraw.Draw(pil)
    h, w = im.shape
    for _ in range(rng.integers(1,3)):
        x0, y0 = int(rng.integers(0, w*0.7)), int(rng.integers(0, h*0.7))
        x1, y1 = x0 + int(rng.integers(w*0.1, w*0.3)), y0 + int(rng.integers(h*0.1, h*0.3))
        val = int(rng.integers(10, 230))
        draw.ellipse([x0, y0, x1, y1], fill=val)
    return np.array(pil)

def make_synth(root="data/synth", n_train=30, n_test_good=20, n_test_anom=20, size=(128,128), seed=42):
    rng = np.random.default_rng(seed)
    paths = {
        "train_good": os.path.join(root, "train", "good"),
        "test_good": os.path.join(root, "test", "good"),
        "test_anom": os.path.join(root, "test", "anomaly"),
        "masks": os.path.join(root, "ground_truth"),
    }
    for p in paths.values(): ensure_dir(p)

    # Train (good)
    for i in range(n_train):
        im = _noise_texture(*size, seed=rng.integers(1e9))
        Image.fromarray(im).save(os.path.join(paths["train_good"], f"{i:04d}.png"))

    # Test good
    for i in range(n_test_good):
        im = _noise_texture(*size, seed=rng.integers(1e9))
        Image.fromarray(im).save(os.path.join(paths["test_good"], f"{i:04d}.png"))

    # Test anomaly + masks
    for i in range(n_test_anom):
        base = _noise_texture(*size, seed=rng.integers(1e9))
        anom = _inject_defect(base, seed=rng.integers(1e9))
        Image.fromarray(anom).save(os.path.join(paths["test_anom"], f"{i:04d}.png"))
        diff = (np.abs(anom.astype(int) - base.astype(int)) > 15).astype(np.uint8)*255
        Image.fromarray(diff).save(os.path.join(paths["masks"], f"{i:04d}_mask.png"))

    return root

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--make-synth", type=str, default=None, help="Output root for synthetic dataset")
    ap.add_argument("--h", type=int, default=128)
    ap.add_argument("--w", type=int, default=128)
    args = ap.parse_args()
    if args.make_synth:
        root = make_synth(args.make_synth, size=(args.h,args.w))
        print("Wrote synthetic dataset to:", root)
