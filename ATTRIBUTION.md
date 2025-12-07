# Attribution

## AI-Generated Code Conversion

The following components were converted from the notebooks with AI assistance (GitHub Copilot and ChatGPT-5):

- `run_pipeline.py`: Pipeline orchestration
- `src/ocr_labels.py`: OCR processing (Tesseract & EasyOCR)
- `src/matcher.py`: Label-to-mask matching
- `src/add_unmatched_masks.py`: SAM mask generation
- `src/utils.py`: Utilities
- `src/sam_masks.py`: Automatic mask genertion for a floorplan image

## AI-Generated Code

The following components were generated with the AI assistance (Gemini 2.5 Pro and ChatGPT-5)

- `ocr.ipynb`
- `index.html`: UI file

## Open Source

The following components were adapted from open-source code published by Meta.

- **SAM 2.1 (Meta)**: Segment Anything Model for mask generation
- `automatic_mask_generator.ipynb`
- `predictor_by_point_and_bbox`
