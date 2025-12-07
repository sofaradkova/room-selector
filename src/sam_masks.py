# AI attrbution: converted by Copilot and ChatGPT-5 from the notebooks
import torch
import numpy as np
from PIL import Image
from pathlib import Path

from sam2.build_sam import build_sam2
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator


def get_default_device():
    if torch.backends.mps.is_available():
        print("Using MPS")
        return torch.device("mps")

    print("Using CPU")
    return torch.device("cpu")


def load_sam_model(checkpoint_path, config_path, device=None):
    device = device or get_default_device()

    checkpoint_path = str(checkpoint_path)
    config_path = str(config_path)

    print("Loading SAM2 model...")
    sam2 = build_sam2(
        config_path,
        checkpoint_path,
        device=device,
        apply_postprocessing=False
    )
    return sam2


def generate_masks(
    image_path,
    checkpoint_path,
    config_path,
    points_per_side=64,
    pred_iou_thresh=0.85,
    stability_score_thresh=0.92,
    min_area=11000,
    max_area=30000,
    max_width=500,
    min_height=100,
    max_height=200,
    device=None,
):
    device = device or get_default_device()

    sam2 = load_sam_model(checkpoint_path, config_path, device=device)

    mask_generator = SAM2AutomaticMaskGenerator(
        model=sam2,
        points_per_side=points_per_side,
        pred_iou_thresh=pred_iou_thresh,
        stability_score_thresh=stability_score_thresh,
        crop_n_layers=1,
        min_mask_region_area=25,
        use_m2m=True,
    )

    image = np.array(Image.open(image_path).convert("RGB"))
    print("Running SAM2 mask generator...")

    masks = mask_generator.generate(image)
    print(f"Generated {len(masks)} raw masks.")

    filtered = []
    for m in masks:
        _, _, w, h = m["bbox"]
        area = m["area"]

        if w <= max_width and min_height <= h <= max_height:
            if min_area <= area <= max_area:
                filtered.append(m)

    print(f"Filtered down to {len(filtered)} masks.")

    return filtered


def save_masks_pickle(masks, out_path):
    import pickle
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, "wb") as f:
        pickle.dump(masks, f, protocol=pickle.HIGHEST_PROTOCOL)

    print("Saved filtered masks ->", out_path)
