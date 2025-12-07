# AI attrbution: converted by Copilot and ChatGPT-5 from the notebooks
import os
import pickle
import json
from typing import List, Tuple, Optional
import numpy as np
from PIL import Image
import torch

from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

def parse_bbox_str(s: str) -> Tuple[int, int, int, int]:
    """Parse a bbox string like '1081,1561,59,13' or '1081 1561 59 13' into ints (x,y,w,h)."""
    parts = s.replace(",", " ").split()
    if len(parts) != 4:
        raise ValueError(f"Invalid bbox spec: {s}")
    return tuple(int(round(float(p))) for p in parts)


def expand_box(bbox: Tuple[int, int, int, int], pad_px: int, image_shape: Tuple[int, int]) -> List[int]:
    """
    bbox: (x,y,w,h)
    image_shape: numpy image shape (H,W,...) or (H,W)
    returns: [x0,y0,x1,y1] clipped to image
    """
    x, y, w, h = bbox
    H, W = image_shape[:2]
    x0 = max(0, int(round(x - pad_px)))
    y0 = max(0, int(round(y - pad_px)))
    x1 = min(W, int(round(x + w + pad_px)))
    y1 = min(H, int(round(y + h + pad_px)))
    return [x0, y0, x1, y1]


def safe_get_bbox(item) -> Optional[Tuple[int, int, int, int]]:
    """Given an element from combined, return bbox (x,y,w,h) or None."""
    try:
        b = item.get("bbox", None)
    except Exception:
        b = None
    if b is None:
        return None
    # accept [x,y,w,h] or tuple
    if len(b) != 4:
        return None
    return tuple(int(round(float(x))) for x in b)


def device_from_args(force_cpu: bool = False):
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
    if force_cpu:
        return torch.device("cpu")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

def main(
    image_path: str,
    combined_path: Optional[str],
    unmatched_indices: Optional[List[int]],
    unmatched_bboxes: Optional[List[Tuple[int, int, int, int]]],
    sam_checkpoint: str,
    sam_cfg: str,
    pad_px: int = 60,
    out_path: str = "masks_added.pkl",
    append_to: Optional[str] = None,
    device_force_cpu: bool = False,
):
    # Load image
    pil = Image.open(image_path)
    image = np.array(pil.convert("RGB"))
    H, W = image.shape[:2]
    print(f"Loaded image {image_path} shape={image.shape}")

    # Load combined list if provided
    combined = None
    if combined_path:
        with open(combined_path, "r", encoding="utf-8") as f:
            combined = json.load(f)
        print(f"Loaded combined list from {combined_path} ({len(combined)} entries)")

    # Convert unmatched indices into bboxes using combined
    bboxes_to_query = []
    if unmatched_indices:
        if combined is None:
            raise ValueError("If passing unmatched_indices, you must supply --combined_json")
        for idx in unmatched_indices:
            if idx < 0 or idx >= len(combined):
                print(f"Warning: index {idx} out of range for combined, skipping")
                continue
            bb = safe_get_bbox(combined[idx])
            if bb is None:
                print(f"Warning: combined[{idx}] has no valid bbox, skipping")
                continue
            bboxes_to_query.append(bb)
    if unmatched_bboxes:
        bboxes_to_query.extend(unmatched_bboxes)

    if not bboxes_to_query:
        print("No unmatched bboxes found / provided. Exiting.")
        return

    # prepare device and model
    device = device_from_args(device_force_cpu)
    print(f"Using device: {device}")
    sam_model = build_sam2(sam_cfg, sam_checkpoint, device=device)
    predictor = SAM2ImagePredictor(sam_model)
    predictor.set_image(image)

    # Optionally load existing `filtered_masks` to append to
    filtered_masks = []
    if append_to and os.path.exists(append_to):
        try:
            with open(append_to, "rb") as f:
                filtered_masks = pickle.load(f)
            print(f"Loaded {len(filtered_masks)} existing masks from {append_to}")
        except Exception as e:
            print(f"Could not load existing masks from {append_to}: {e}")

    new_mask_dicts = []
    for i, bbox in enumerate(bboxes_to_query):
        print(f"Querying SAM for bbox #{i}: {bbox}")
        input_box = np.array(expand_box(bbox, pad_px, image.shape))[None, :]  # shape (1,4)
        masks, scores, _ = predictor.predict(
            point_coords=None,
            point_labels=None,
            box=input_box,
            multimask_output=False,
        )
        # masks: numpy array like (N, H, W) or (1, H, W) depending on predictor
        if masks is None:
            print(f"Predictor returned no masks for bbox {bbox}")
            continue

        masks = np.asarray(masks)
        if masks.ndim == 2:
            masks = masks[None, ...]  # make (1,H,W)

        for j in range(masks.shape[0]):
            seg = np.squeeze(masks[j]).astype(bool)
            ys, xs = np.where(seg)
            if len(xs) == 0:
                print(f" - mask {j} is empty, skipping")
                continue
            x_min, x_max = int(xs.min()), int(xs.max())
            y_min, y_max = int(ys.min()), int(ys.max())
            bbox_new = [x_min, y_min, int(x_max - x_min), int(y_max - y_min)]
            area = int(seg.sum())

            # predicted iou: try to read from scores if present
            iou = 1.0
            try:
                if scores is not None:
                    # scores might be shape (N,) or list
                    s_val = np.asarray(scores)[j]
                    iou = float(s_val)
            except Exception:
                pass

            pc = None
            stability = 1.0
            crop_box = [0, 0, seg.shape[1], seg.shape[0]]

            md = {
                "segmentation": seg,
                "area": area,
                "bbox": bbox_new,
                "predicted_iou": iou,
                "point_coords": pc,
                "stability_score": stability,
                "crop_box": crop_box,
            }
            new_mask_dicts.append(md)
            print(f" - added mask: bbox={bbox_new}, area={area}, iou={iou:.3f}")

    # Append and save
    filtered_masks.extend(new_mask_dicts)
    print(f"Total new masks added: {len(new_mask_dicts)}. Combined count: {len(filtered_masks)}")

    with open(out_path, "wb") as f:
        pickle.dump(filtered_masks, f)
    print(f"Wrote appended masks to {out_path}")