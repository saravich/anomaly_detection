import os, numpy as np, pickle
import cv2 as cv
from skimage.feature import local_binary_pattern
from skimage.metrics import structural_similarity as ssim
from sklearn.ensemble import IsolationForest
from .utils import imread_gray, resize, ensure_dir

# SSIM Template

def build_template(train_paths, size_hw):
    ims = [resize(imread_gray(p), size_hw) for p in train_paths]
    ref = np.median(np.stack(ims, 0), axis=0).astype(np.uint8)
    return ref

def ssim_map(im, ref):
    im = im.astype(np.float32)
    ref = ref.astype(np.float32)
    score, m = ssim(im, ref, data_range=255.0, full=True)
    return 1.0 - m.astype(np.float32), float(1.0 - score)

# LBP + IsolationForest

def _lbp_hist(im):
    lbp = local_binary_pattern(im, 8, 1, 'uniform')
    hist, _ = np.histogram(lbp, bins=np.arange(0,60), density=True)
    return hist

def lbp_iforest_fit(train_paths, size_hw):
    feats = []
    for p in train_paths:
        im = resize(imread_gray(p), size_hw)
        feats.append(_lbp_hist(im))
    clf = IsolationForest(contamination='auto', random_state=0).fit(feats)
    return clf

def lbp_iforest_score(model, im):
    hist = _lbp_hist(im)
    return -float(model.score_samples([hist])[0])

# Model Save & Load

def save_model(model, path):
    ensure_dir(os.path.dirname(path) or ".")
    with open(path, "wb") as f:
        pickle.dump(model, f)

def load_model(path):
    with open(path, "rb") as f:
        return pickle.load(f)
