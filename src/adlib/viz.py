import numpy as np, cv2 as cv

def overlay_heatmap(gray, heat):
    gray3 = cv.cvtColor(gray, cv.COLOR_GRAY2BGR)
    heat_norm = (255 * (heat - heat.min()) / (1e-6 + heat.max() - heat.min())).astype(np.uint8)
    heat_color = cv.applyColorMap(heat_norm, cv.COLORMAP_JET)
    out = cv.addWeighted(gray3, 0.6, heat_color, 0.6, 0)
    return out
