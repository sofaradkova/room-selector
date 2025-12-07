import pickle
from pathlib import Path
import numpy as np
from PIL import Image

def save_masks_pickle(masks, path):
    """Save list of mask dicts to a pickle."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'wb') as f:
        pickle.dump(masks, f, protocol=pickle.HIGHEST_PROTOCOL)
    print("Saved masks pickle ->", path)


def load_masks_pickle(path):
    """Load mask pickle safely."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Mask pickle not found: {path}")
    with open(path, 'rb') as f:
        masks = pickle.load(f)
    print("Loaded", len(masks), "masks from", path)
    return masks

def bbox_contains_point(bbox, point):
    """
    bbox: (x,y,w,h)
    point: (px,py)
    """
    x, y, w, h = bbox
    px, py = point
    return (px >= x) and (px <= x + w) and (py >= y) and (py <= y + h)

def mask_bbox_from_bool(mask_bool):
    """Returns (x,y,w,h) for a 2D boolean mask"""
    ys, xs = np.where(mask_bool)
    if len(xs) == 0:
        return (0, 0, 0, 0)
    x_min, x_max = xs.min(), xs.max()
    y_min, y_max = ys.min(), ys.max()
    return (int(x_min), int(y_min), int(x_max - x_min), int(y_max - y_min))
