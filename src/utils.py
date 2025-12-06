import json, pickle
from pathlib import Path
import numpy as np
from PIL import Image


# -----------------------------------------------------------
# BASIC I/O HELPERS
# -----------------------------------------------------------

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


def save_mask_png(mask_bool, path):
    """
    Save a boolean mask (HxW) as a PNG, converting to uint8.
    """
    arr = np.asarray(mask_bool).astype('uint8')
    arr = arr * 255  # True=255
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(arr).save(path)
    print("Saved mask PNG ->", path)


def save_labels_json(labels, path):
    """
    Save labels JSON in your preferred format:
    {
        "base_image": "floorplan6.png",
        "labels": [...]
    }
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    out = {
        "base_image": "floorplan6.png",
        "labels": labels
    }

    # ensure centroids JSON-serializable (floats)
    for l in out["labels"]:
        if "centroid" in l:
            cx, cy = l["centroid"]
            l["centroid"] = [float(cx), float(cy)]

    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print("Saved labels JSON ->", path)


# -----------------------------------------------------------
# GEOMETRIC UTILITIES NEEDED BY MATCHER + OCR
# -----------------------------------------------------------

def bbox_contains_point(bbox, point):
    """
    bbox: (x,y,w,h)
    point: (px,py)
    """
    x, y, w, h = bbox
    px, py = point
    return (px >= x) and (px <= x + w) and (py >= y) and (py <= y + h)


def _mask_bbox_from_seg(seg):
    """
    Compute bbox (x,y,w,h) from mask segmentation.
    Supports:
      - boolean numpy array
      - polygon list (flat or nested)
    """
    if seg is None:
        return (0, 0, 0, 0)

    arr = np.asarray(seg)

    # Case 1: boolean HxW mask
    if arr.ndim == 2:
        ys, xs = np.nonzero(arr)
        if xs.size == 0:
            return (0, 0, 0, 0)
        x0, x1 = xs.min(), xs.max()
        y0, y1 = ys.min(), ys.max()
        return (int(x0), int(y0), int(x1 - x0 + 1), int(y1 - y0 + 1))

    # Case 2: polygon list-of-points or flattened numeric
    if isinstance(seg, (list, tuple)) and len(seg) > 0:
        if isinstance(seg[0], (int, float)):
            pts = [(seg[i], seg[i+1]) for i in range(0, len(seg), 2)]
        else:
            pts = seg
        xs = [int(p[0]) for p in pts]
        ys = [int(p[1]) for p in pts]
        if xs:
            x0, x1 = min(xs), max(xs)
            y0, y1 = min(ys), max(ys)
            return (x0, y0, x1 - x0 + 1, y1 - y0 + 1)

    return (0, 0, 0, 0)
