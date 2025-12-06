#!/usr/bin/env python3
"""
clean_and_export_masks.py

- Loads filtered_masks.pkl
- Converts each mask to rectangle mask
- Recomputes bbox + area
- Re-matches labels using one_to_one_match_by_centroid
- Saves mask_N.png + labels.json
"""

import os
import json
import pickle
import numpy as np
from PIL import Image

# -------------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------------
def mask_bbox_from_bool(mask_bool):
    """Returns (x,y,w,h) for a 2D boolean mask"""
    ys, xs = np.where(mask_bool)
    if len(xs) == 0:
        return (0, 0, 0, 0)
    x_min, x_max = xs.min(), xs.max()
    y_min, y_max = ys.min(), ys.max()
    return (int(x_min), int(y_min), int(x_max - x_min), int(y_max - y_min))


def create_rectangular_mask(mask_bool):
    """
    Creates a new boolean mask representing the tightest axis-aligned bounding box
    of the input irregular mask.
    """
    H, W = mask_bool.shape
    x, y, w, h = mask_bbox_from_bool(mask_bool)

    rectangular_mask = np.full((H, W), False, dtype=bool)

    # Bound clipping
    x1 = max(0, x)
    y1 = max(0, y)
    x2 = min(W, x + w)
    y2 = min(H, y + h)

    if x2 > x1 and y2 > y1:
        rectangular_mask[y1:y2, x1:x2] = True

    return rectangular_mask


# -------------------------------------------------------------------------
# MAIN PIPELINE
# -------------------------------------------------------------------------
def main():

    # --- CONFIG ---
    FILTERED_MASKS_PKL = "filtered_masks.pkl"  # <-- your masks file
    COMBINED_JSON      = "combined.json"       # <-- your OCR labels file
    OUTPUT_DIR         = "output"              # <-- final output folder
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ---------------------------------------------------------------------
    # Load inputs
    # ---------------------------------------------------------------------
    print("Loading filtered masks...")
    with open(FILTERED_MASKS_PKL, "rb") as f:
        filtered_masks = pickle.load(f)

    print(f"Loaded {len(filtered_masks)} masks.")

    print("Loading combined labels...")
    with open(COMBINED_JSON, "r") as f:
        combined = json.load(f)


    # ---------------------------------------------------------------------
    # STEP 1 — Clean up masks (rectangular masks)
    # ---------------------------------------------------------------------
    print("Rectangularizing masks...")

    for mask_dict in filtered_masks:
        seg_bool = mask_dict["segmentation"]

        rect_seg = create_rectangular_mask(seg_bool)
        mask_dict["segmentation"] = rect_seg
        mask_dict["area"] = int(rect_seg.sum())
        mask_dict["bbox"] = mask_bbox_from_bool(rect_seg)

    print("Rectangular mask cleanup complete.")


    # ---------------------------------------------------------------------
    # STEP 2 — Re-match labels and masks
    # ---------------------------------------------------------------------
    print("Running centroid matching...")

    # You already have this function in your notebook
    from matcher import one_to_one_match_by_centroid  # EDIT to match your code

    matched_labels, unmatched_labels, unmatched_masks = \
        one_to_one_match_by_centroid(combined, filtered_masks)

    print(f"Matched: {len(matched_labels)}")
    print(f"Unmatched labels: {len(unmatched_labels)}")
    print(f"Unmatched masks: {len(unmatched_masks)}")


    # ---------------------------------------------------------------------
    # STEP 3 — Export masks as mask_N.png + labels.json
    # ---------------------------------------------------------------------
    print("Saving mask PNGs + labels.json...")

    labels_out = []

    for i, lab in enumerate(matched_labels):

        seg = lab.get("mask")
        txt = lab.get("text", f"label_{i}")
        mask_id = lab.get("mask_id")

        if seg is not None:
            arr = np.asarray(seg).astype(bool)
            png = (arr.astype(np.uint8) * 255)
            im = Image.fromarray(png)

            fname = f"mask_{i}.png"
            im.save(os.path.join(OUTPUT_DIR, fname))
        else:
            fname = None

        labels_out.append({
            "index": i,
            "text": txt,
            "mask_id": mask_id,
            "mask_file": fname
        })

    # Save manifest
    out_json = {
        "base_image": "floorplan6.png",
        "labels": labels_out
    }

    with open(os.path.join(OUTPUT_DIR, "labels.json"), "w", encoding="utf-8") as f:
        json.dump(out_json, f, indent=2)

    print(f"Done. Masks + labels.json saved in: {OUTPUT_DIR}")


# -------------------------------------------------------------------------
if __name__ == "__main__":
    main()
