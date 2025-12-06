import json, pickle
from pathlib import Path
import numpy as np
from PIL import Image

def save_masks_pickle(masks, path):
    with open(path, 'wb') as f: pickle.dump(masks, f)

def load_masks_pickle(path):
    with open(path, 'rb') as f: return pickle.load(f)

def save_mask_png(mask_bool, path):
    arr = (mask_bool.astype('uint8') * 255)
    Image.fromarray(arr).save(path)

def save_labels_json(labels, path):
    Path(path).parent.mkdir(parents=True,exist_ok=True)
    with open(path,'w') as f: json.dump({"base_image": "floorplan6.png", "labels": labels}, f, indent=2)
