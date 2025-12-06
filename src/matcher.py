from typing import List, Dict, Any, Tuple
import numpy as np

from src.utils import bbox_contains_point

def _centroid_for_label(lbl: Dict[str, Any]) -> Tuple[float, float]:
    """Return (cx, cy) for a label dict that may have 'centroid' or 'cx'/'cy' or 'bbox'."""
    if "centroid" in lbl:
        c = lbl["centroid"]
        return (float(c[0]), float(c[1]))
    if "cx" in lbl and "cy" in lbl:
        return (float(lbl["cx"]), float(lbl["cy"]))
    if "bbox" in lbl:
        x, y, w, h = lbl["bbox"]
        return (float(x + w / 2.0), float(y + h / 2.0))
    raise ValueError("Label missing centroid or bbox.")


def _mask_bbox_from_seg(seg) -> Tuple[int, int, int, int]:
    """Derive bbox (x,y,w,h) from segmentation boolean array or polygon-like data."""
    if seg is None:
        return (0, 0, 0, 0)
    arr = np.asarray(seg)
    if arr.ndim == 2:
        ys, xs = np.nonzero(arr)
        if xs.size == 0:
            return (0, 0, 0, 0)
        x0, x1 = int(xs.min()), int(xs.max())
        y0, y1 = int(ys.min()), int(ys.max())
        return (x0, y0, int(x1 - x0 + 1), int(y1 - y0 + 1))
    # polygon-ish: try flatten to pairs
    if isinstance(seg, (list, tuple)) and len(seg) > 0:
        if isinstance(seg[0], (int, float)):
            pts = [(seg[i], seg[i+1]) for i in range(0, len(seg), 2)]
        else:
            pts = seg
        xs = [int(round(p[0])) for p in pts]
        ys = [int(round(p[1])) for p in pts]
        if not xs:
            return (0, 0, 0, 0)
        x0, x1 = min(xs), max(xs)
        y0, y1 = min(ys), max(ys)
        return (x0, y0, int(x1 - x0 + 1), int(y1 - y0 + 1))
    return (0, 0, 0, 0)


def match_labels_to_masks(
    ocr_labels: List[Dict[str, Any]],
    masks: List[Dict[str, Any]]
) -> Tuple[List[Dict[str, int]], List[int], List[int]]:
    """
    Strict containment matcher.

    Returns:
      matches: list of {"label_index": int, "mask_index": int}
      unmatched_label_indices: list[int]
      unmatched_mask_indices: list[int]
    """
    nL = len(ocr_labels)
    nM = len(masks)

    # compute label centroids
    label_centroids = []
    for li, lbl in enumerate(ocr_labels):
        cx, cy = _centroid_for_label(lbl)
        label_centroids.append((cx, cy))

    # compute mask bboxes (ensure bbox exists)
    mask_bboxes = []
    for mi, m in enumerate(masks):
        bbox = m.get("bbox", None)
        if bbox is None:
            seg = m.get("segmentation", None) or m.get("mask", None)
            if seg is not None:
                bbox = _mask_bbox_from_seg(seg)
            else:
                bbox = (0, 0, 0, 0)
        mask_bboxes.append(tuple(bbox))

    # For each mask, collect labels whose centroid is inside its bbox
    mask_to_label_candidates = {mi: [] for mi in range(nM)}
    for li, (cx, cy) in enumerate(label_centroids):
        for mi, bbox in enumerate(mask_bboxes):
            if bbox is None:
                continue
            if bbox_contains_point(bbox, (cx, cy)):
                mask_to_label_candidates[mi].append(li)

    # Now pick one label per mask. If multiple candidates, pick highest confidence (fallback to lower index)
    label_assigned_to_mask = {}  # mask_index -> label_index
    label_taken = set()

    for mi in range(nM):
        cands = mask_to_label_candidates.get(mi, [])
        if not cands:
            continue
        # pick best by confidence
        def conf_of(li):
            conf = ocr_labels[li].get("confidence", None)
            try:
                return float(conf) if conf is not None else -1.0
            except:
                return -1.0
        # sort candidates by (-confidence, label_index) and choose first that's not already taken
        sorted_cands = sorted(cands, key=lambda li: (-conf_of(li), li))
        chosen = None
        for li in sorted_cands:
            if li not in label_taken:
                chosen = li
                break
        if chosen is not None:
            label_assigned_to_mask[mi] = chosen
            label_taken.add(chosen)

    # Build matches list and unmatched lists
    matches = []
    for mi, li in label_assigned_to_mask.items():
        matches.append({"label_index": int(li), "mask_index": int(mi)})

    unmatched_label_indices = [i for i in range(nL) if i not in label_taken]
    unmatched_mask_indices = [i for i in range(nM) if i not in label_assigned_to_mask]

    # sort for determinism
    matches.sort(key=lambda x: x["label_index"])
    unmatched_label_indices.sort()
    unmatched_mask_indices.sort()

    return matches, unmatched_label_indices, unmatched_mask_indices


def produce_labels_with_mask_index(
    ocr_labels: List[Dict[str, Any]],
    masks: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """
    Return shallow copies of ocr_labels with an added key 'mask_index' (int or None).
    """
    matches, unmatched_labels, unmatched_masks = match_labels_to_masks(ocr_labels, masks)
    label_to_mask = {m["label_index"]: m["mask_index"] for m in matches}
    out = []
    for li, lbl in enumerate(ocr_labels):
        entry = dict(lbl)
        entry["mask_index"] = label_to_mask.get(li, None)
        out.append(entry)
    return out