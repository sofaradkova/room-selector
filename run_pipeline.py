import os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
import logging
from pathlib import Path
from src.sam_masks import generate_masks, save_masks_pickle
from src.ocr_labels import run_ocr_on_image
#from src.matcher import one_to_one_match_by_centroid
#from src.box_filler import fill_missing_masks
from src.utils import save_labels_json, save_mask_png, load_masks_pickle

logging.basicConfig(level=logging.INFO)
BASE = Path("data")
RAW = BASE/"raw"
INTER = BASE/"intermediate"
OUT = BASE/"outputs"
INTER.mkdir(parents=True, exist_ok=True)
OUT.mkdir(parents=True, exist_ok=True)

def main(image_path: str):
    image_path = "data/raw/floorplan6.png"
    checkpoint_path = "checkpoints/sam2.1_hiera_large.pt"
    config_path = "configs/sam2.1_hiera_l.yaml"

    masks = generate_masks(image_path, checkpoint_path, config_path)

    save_masks_pickle(masks, "data/output/filtered_masks.pkl")

if __name__ == '__main__':
    import sys, json
    if len(sys.argv) < 2:
        print("Usage: python run_pipeline.py data/raw/floorplan6.png")
        sys.exit(1)
    main(sys.argv[1])
