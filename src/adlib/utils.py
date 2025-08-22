import os, json

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def save_json(obj, path: str):
    ensure_dir(os.path.dirname(path) or ".")
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)

def load_json(path: str):
    with open(path, "r") as f:
        return json.load(f)

def list_images(root: str):
    exts = (".png", ".jpg", ".jpeg", ".bmp")
    paths = []
    for dp, _, files in os.walk(root):
        for fn in files:
            if fn.lower().endswith(exts):
                paths.append(os.path.join(dp, fn))
    return sorted(paths)

def imread_gray(path: str):
    import cv2 as cv
    im = cv.imread(path, cv.IMREAD_GRAYSCALE)
    if im is None:
        raise FileNotFoundError(path)
    return im

def resize(im, size_hw):
    import cv2 as cv
    h, w = size_hw
    return cv.resize(im, (w, h), interpolation=cv.INTER_AREA)
