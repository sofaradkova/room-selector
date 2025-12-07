from typing import List, Dict, Tuple, Optional
import re
import math

import cv2
import numpy as np
import pytesseract
from pytesseract import Output
import easyocr


DEFAULT_PATTERN = re.compile(r'^[A-Z]\d{4}[A-Z]$', re.IGNORECASE)


def _clahe_enhance_gray(gray: np.ndarray, clipLimit: float = 2.0, tileGridSize: Tuple[int, int] = (8, 8)) -> np.ndarray:
    """Apply CLAHE to a grayscale image and return enhanced gray image (uint8)."""
    clahe = cv2.createCLAHE(clipLimit=clipLimit, tileGridSize=tileGridSize)
    enhanced = clahe.apply(gray)
    return enhanced


def ocr_image_tesseract(image_path_or_array,
                        tess_config: str = "--psm 12",
                        pattern: Optional[re.Pattern] = None,
                        min_confidence: float = None) -> List[Dict]:
    """
    Run pytesseract on the image (path or numpy array).
    If min_confidence is set, entries with confidence < min_confidence are excluded.
    """
    # load image (same as you already have)...
    if isinstance(image_path_or_array, str):
        img = cv2.imread(image_path_or_array, cv2.IMREAD_COLOR)
    else:
        img = image_path_or_array.copy()

    if img is None:
        raise FileNotFoundError(f"Image not found or cannot be read: {image_path_or_array}")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    enhanced = _clahe_enhance_gray(gray)
    ocr_data = pytesseract.image_to_data(enhanced, output_type=Output.DICT, config=tess_config)

    out = []
    n = len(ocr_data.get('text', []))
    for i in range(n):
        raw_text = (ocr_data['text'][i] or "").strip()
        if not raw_text:
            continue

        # parse confidence robustly
        conf_raw = ocr_data.get('conf', [None]*n)[i]
        try:
            conf = float(conf_raw)
        except Exception:
            conf = -1.0

        # optional confidence threshold
        if (min_confidence is not None) and (conf < float(min_confidence)):
            continue

        # optional pattern filtering
        if pattern is not None and not pattern.match(raw_text):
            continue

        # safe bbox parsing: keep None if missing instead of skipping the entry
        def _safe_int_field(key, idx):
            try:
                val = ocr_data.get(key, [None]*n)[idx]
                if val is None or val == '':
                    return None
                return int(float(val))
            except Exception:
                return None

        x = _safe_int_field('left', i)
        y = _safe_int_field('top', i)
        w = _safe_int_field('width', i)
        h = _safe_int_field('height', i)

        # compute center if bbox present, otherwise None
        cx = (x + w / 2.0) if (x is not None and w is not None) else None
        cy = (y + h / 2.0) if (y is not None and h is not None) else None

        out.append({
            "text": raw_text,
            "confidence": float(conf),
            "bbox": (x, y, w, h),
            "cx": cx,
            "cy": cy,
            "source": "tesseract"
        })
    return out


def ocr_image_easyocr(image_path_or_array,
                      pattern: Optional[re.Pattern] = None,
                      reader: Optional["easyocr.Reader"] = None) -> List[Dict]:
    """
    Run EasyOCR on the image (path or numpy array). Returns list of dicts:
      {'text','confidence' (0..100),'bbox':(x,y,w,h),'cx','cy','source':'easyocr'}
    If easyocr is not available, raises ImportError.
    """
    # load image
    if isinstance(image_path_or_array, str):
        img = cv2.imread(image_path_or_array, cv2.IMREAD_COLOR)
    else:
        img = image_path_or_array.copy()

    if img is None:
        raise FileNotFoundError(f"Image not found or cannot be read: {image_path_or_array}")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    enhanced = _clahe_enhance_gray(gray)

    if reader is None:
        # create a reader (lazy). GPU usage is optional; default to False for compatibility.
        reader = easyocr.Reader(['en'], gpu=False)

    # EasyOCR expects RGB arrays
    rgb = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2RGB)
    results = reader.readtext(rgb, paragraph=False)

    out = []
    for bbox_pts, text_raw, conf_raw in results:
        if not text_raw:
            continue
        text = text_raw.strip().upper()  # normalize to uppercase before pattern match
        if pattern is not None and not pattern.match(text):
            continue
        # bbox_pts are four corner points; convert to rectangle
        xs = [int(round(p[0])) for p in bbox_pts]
        ys = [int(round(p[1])) for p in bbox_pts]
        x0, y0 = min(xs), min(ys)
        w, h = max(xs) - x0, max(ys) - y0
        cx = x0 + w / 2.0; cy = y0 + h / 2.0
        # Confidence normalization: EasyOCR often returns 0..1 but sometimes other ranges; normalize to 0..100 heuristically
        try:
            conf = float(conf_raw)
            conf100 = conf * 100.0 if conf <= 1.0 else conf
        except Exception:
            conf100 = 0.0
        out.append({
            "text": text,
            "confidence": float(conf100),
            "bbox": (x0, y0, w, h),
            "cx": float(cx),
            "cy": float(cy),
            "source": "easyocr"
        })
    return out


def iou_box(a: Tuple[float, float, float, float],
            b: Tuple[float, float, float, float]) -> float:
    """
    Compute IoU between two axis-aligned boxes in (x, y, w, h) format.
    Returns IoU in [0,1].
    """
    ax, ay, aw, ah = a
    bx, by, bw, bh = b

    x1 = max(ax, bx)
    y1 = max(ay, by)
    x2 = min(ax + aw, bx + bw)
    y2 = min(ay + ah, by + bh)

    iw = max(0.0, x2 - x1)
    ih = max(0.0, y2 - y1)
    inter = iw * ih

    area_a = max(0.0, aw) * max(0.0, ah)
    area_b = max(0.0, bw) * max(0.0, bh)
    union = area_a + area_b - inter

    if union <= 0.0:
        return 0.0
    return float(inter) / float(union)


def _centroid_from_bbox(box: Tuple[int, int, int, int]) -> Tuple[float, float]:
    x, y, w, h = box
    return (x + w / 2.0, y + h / 2.0)


def _dist(p: Tuple[float, float], q: Tuple[float, float]) -> float:
    return math.hypot(p[0] - q[0], p[1] - q[1])


def combine_ocr_results(tess_list: List[Dict],
                        easy_list: List[Dict],
                        iou_thresh: float = 0.25,
                        dist_thresh: float = 30.0) -> List[Dict]:
    """
    Combine pytesseract list (tess_list) and easyocr list (easy_list).

    Strategy:
      - Keep all tess items (higher precedence).
      - For each easy item, add it only if it does not overlap (IoU >= iou_thresh)
        nor is it too close (centroid distance <= dist_thresh) to any tess item.

    Returns combined list (tess items first, new easy items appended).
    """
    tess_boxes = [t['bbox'] for t in tess_list]
    tess_centroids = [(t.get('cx') or _centroid_from_bbox(t['bbox']),
                       t.get('cy') or _centroid_from_bbox(t['bbox'])[1]) for t in tess_list]
    # normalize tess_centroids to tuples
    tess_centroids = [(c[0], c[1]) if isinstance(c, tuple) else (float(c), float(c)) for c in tess_centroids]

    added = []
    for item in easy_list:
        rect = item['bbox']
        cx = item['cx']; cy = item['cy']
        too_close = False
        for tb, tc in zip(tess_boxes, tess_centroids):
            if iou_box(tb, rect) >= iou_thresh:
                too_close = True
                break
            if _dist((cx, cy), tc) <= dist_thresh:
                too_close = True
                break
        if not too_close:
            added.append(item)

    combined = list(tess_list) + added
    return combined

def run_ocr_combined(image_path: str,
                     pattern: Optional[re.Pattern] = DEFAULT_PATTERN,
                     tess_config: str = "--psm 12",
                     easy_iou_thresh: float = 0.25,
                     easy_dist_thresh: float = 30.0,
                     easy_reader: Optional["easyocr.Reader"] = None) -> List[Dict]:
    """
    High-level convenience function: runs pytesseract (and optionally EasyOCR),
    merges results, normalizes text, and returns per-label dicts with keys:
      'text', 'confidence', 'bbox', 'centroid', 'source'
    """
    tlist = ocr_image_tesseract(image_path, tess_config, pattern=pattern)

    elist = ocr_image_easyocr(image_path, pattern=pattern, reader=easy_reader)

    combined = combine_ocr_results(tlist, elist, iou_thresh=easy_iou_thresh, dist_thresh=easy_dist_thresh)

    out = []
    for item in combined:
        # pick centroid either from cx/cy or from bbox
        if 'cx' in item and 'cy' in item:
            cx = float(item['cx']); cy = float(item['cy'])
        else:
            cx, cy = _centroid_from_bbox(item['bbox'])
        out.append({
            "text": item['text'],
            "confidence": float(item.get("confidence", 0.0)),
            "bbox": (int(item['bbox'][0]), int(item['bbox'][1]), int(item['bbox'][2]), int(item['bbox'][3])),
            "centroid": [float(cx), float(cy)],
            "source": item.get("source", "unknown")
        })
    return out