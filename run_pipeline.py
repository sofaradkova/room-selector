import os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
import logging
import json
import pickle
import numpy as np
from pathlib import Path
from PIL import Image

from src.ocr_labels import run_ocr_combined
from src.matcher import match_labels_to_masks
from src.add_unmatched_masks import main as add_unmatched_masks

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

BASE = Path("data")
RAW = BASE/"raw"
INTER = BASE/"intermediate"
OUT = BASE/"output"
INTER.mkdir(parents=True, exist_ok=True)
OUT.mkdir(parents=True, exist_ok=True)


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


def rectangularize_masks(masks):
    """Convert all masks to rectangular masks and recompute bbox + area."""
    logger.info("Rectangularizing masks...")
    for mask_dict in masks:
        seg_bool = mask_dict["segmentation"]
        rect_seg = create_rectangular_mask(seg_bool)
        mask_dict["segmentation"] = rect_seg
        mask_dict["area"] = int(rect_seg.sum())
        mask_dict["bbox"] = mask_bbox_from_bool(rect_seg)
    logger.info("Rectangular mask cleanup complete.")


def export_masks_and_labels(masks, labels, output_dir):
    """Export masks as PNG files and create labels.json manifest."""
    logger.info("Exporting masks and labels...")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    labels_out = []
    export_index = 0

    for i, mask_dict in enumerate(masks):
        seg = mask_dict.get("segmentation")
        
        # Find the label corresponding to this mask (if any)
        label_text = None
        for lbl in labels:
            if lbl.get("mask_index") == i:
                label_text = lbl.get("text", None)
                break

        # Skip unlabeled masks
        if label_text is None:
            logger.info(f"Skipping unlabeled mask at index {i}")
            continue

        if seg is not None:
            arr = np.asarray(seg).astype(bool)
            png = (arr.astype(np.uint8) * 255)
            im = Image.fromarray(png)

            fname = f"mask_{export_index}.png"
            im.save(output_dir / fname)
        else:
            fname = None

        labels_out.append({
            "index": export_index,
            "text": label_text,
            "mask_file": fname
        })
        export_index += 1

    # Save manifest
    out_json = {
        "base_image": "floorplan6.png",
        "labels": labels_out
    }

    with open(output_dir / "labels.json", "w", encoding="utf-8") as f:
        json.dump(out_json, f, indent=2)

    logger.info(f"Exported {len(labels_out)} labeled masks and labels to: {output_dir}")


def main(image_path: str):
    """
    Complete pipeline:
    1. Load pre-computed filtered_masks.pkl
    2. Run both OCRs and combine results
    3. Match labels to masks
    4. Handle unmatched labels with add_unmatched_masks
    5. Rectangularize and export
    """
    image_path = "data/raw/floorplan6.png"
    filtered_masks_path = "data/output/filtered_masks.pkl"
    checkpoint_path = "checkpoints/sam2.1_hiera_large.pt"
    config_path = "configs/sam2.1_hiera_l.yaml"

    # ===================================================================
    # STEP 1: Load filtered masks (pre-computed by SAM)
    # ===================================================================
    logger.info("Loading filtered masks...")
    with open(filtered_masks_path, "rb") as f:
        filtered_masks = pickle.load(f)
    logger.info(f"Loaded {len(filtered_masks)} masks from {filtered_masks_path}")

    # ===================================================================
    # STEP 2: Run both OCRs and combine results
    # ===================================================================
    logger.info("Running OCR pipeline (Tesseract + EasyOCR)...")
    ocr_labels = run_ocr_combined(image_path, use_easyocr=True)
    logger.info(f"Found {len(ocr_labels)} OCR labels")

    # Save combined OCR results for reference
    ocr_json_path = OUT / "combined_ocr.json"
    with open(ocr_json_path, "w", encoding="utf-8") as f:
        # Convert for JSON serialization
        ocr_serializable = []
        for lbl in ocr_labels:
            lbl_copy = dict(lbl)
            if isinstance(lbl_copy.get("bbox"), tuple):
                lbl_copy["bbox"] = list(lbl_copy["bbox"])
            if isinstance(lbl_copy.get("centroid"), (list, tuple)):
                lbl_copy["centroid"] = list(lbl_copy["centroid"])
            ocr_serializable.append(lbl_copy)
        json.dump(ocr_serializable, f, indent=2)
    logger.info(f"Saved OCR results to {ocr_json_path}")

    # ===================================================================
    # STEP 3: Match labels to masks
    # ===================================================================
    logger.info("Running label-to-mask matching...")
    matches, unmatched_label_indices, unmatched_mask_indices = match_labels_to_masks(
        ocr_labels, filtered_masks
    )
    logger.info(f"Matched: {len(matches)} labels to masks")
    logger.info(f"Unmatched labels: {len(unmatched_label_indices)}")
    logger.info(f"Unmatched masks: {len(unmatched_mask_indices)}")

    # Create augmented labels with mask_index
    labels_with_mask_index = []
    for li, lbl in enumerate(ocr_labels):
        entry = dict(lbl)
        # Find mask_index for this label
        mask_idx = None
        for match in matches:
            if match["label_index"] == li:
                mask_idx = match["mask_index"]
                break
        entry["mask_index"] = mask_idx
        labels_with_mask_index.append(entry)

    # ===================================================================
    # STEP 4: Handle unmatched labels with add_unmatched_masks
    # ===================================================================
    if unmatched_label_indices:
        logger.info(f"Generating masks for {len(unmatched_label_indices)} unmatched labels...")
        
        # Prepare unmatched indices for add_unmatched_masks
        unmatched_labels_for_sam = [ocr_labels[i] for i in unmatched_label_indices]
        unmatched_bboxes = [lbl["bbox"] for lbl in unmatched_labels_for_sam]
        
        temp_masks_path = OUT / "temp_added_masks.pkl"
        
        try:
            add_unmatched_masks(
                image_path=image_path,
                combined_path=None,
                unmatched_indices=None,
                unmatched_bboxes=unmatched_bboxes,
                sam_checkpoint=checkpoint_path,
                sam_cfg=config_path,
                pad_px=60,
                out_path=str(temp_masks_path),
                append_to=str(filtered_masks_path),
                device_force_cpu=False,
            )
            
            # Load the combined masks
            with open(temp_masks_path, "rb") as f:
                filtered_masks = pickle.load(f)
            logger.info(f"Combined masks: {len(filtered_masks)} total")
            
            # Clean up temp file
            temp_masks_path.unlink()
        except Exception as e:
            logger.error(f"Error in add_unmatched_masks: {e}")
            logger.info("Continuing with original masks only...")
    else:
        logger.info("All labels matched; no additional mask generation needed.")

    # ===================================================================
    # STEP 5: Rectangularize masks
    # ===================================================================
    rectangularize_masks(filtered_masks)

    # ===================================================================
    # STEP 6: Export final results
    # ===================================================================
    export_masks_and_labels(filtered_masks, labels_with_mask_index, OUT)

    # Save final masks pickle
    final_masks_path = OUT / "final_masks.pkl"
    with open(final_masks_path, "wb") as f:
        pickle.dump(filtered_masks, f)
    logger.info(f"Saved final masks to {final_masks_path}")


if __name__ == '__main__':
    import sys
    if len(sys.argv) < 2:
        print("Usage: python run_pipeline.py data/raw/floorplan6.png")
        sys.exit(1)
    main(sys.argv[1])
