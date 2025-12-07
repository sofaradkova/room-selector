# Setup Instructions

The website is publicly accessible (https://sofaradkova.github.io/room-selector/). To run it locally or to use a different floorplan follow the instructions below.

## Prerequisites

- Conda (Anaconda or Miniconda)
- Tesseract OCR installed (`brew install tesseract` on macOS)

## Installation

1. **Clone the repository**

   ```bash
   git clone git@github.com:sofaradkova/room-selector.git
   cd room-selector
   ```

2. **Create conda environment**

   ```bash
   conda env create -f environment.yml
   conda activate room-selector
   ```

3. **Download SAM 2.1 checkpoint**

   - Download `sam2.1_hiera_large.pt` from [Meta's SAM 2.1 releases](https://github.com/facebookresearch/sam2)
   - Place in `checkpoints/` directory

4. **Ensure SAM 2.1 config is in place**
   - `configs/sam2.1_hiera_l.yaml` should be present (included in repo)

## Running the Pipeline

```bash
python src/run_pipeline.py
```

This will:

1. Load pre-computed SAM masks from `data/output/` (due to SAM model not being fully adapted for MPS, I had to run this part of the pipeline in the Colab notebook but it can be run locally if CUDA is available)
2. Run OCR (Tesseract & EasyOCR) on the floor plan
3. Match OCR labels to masks
4. Pass unmatched labels to SAM to generate remaining masks one-by-one
5. Rectangularize all masks
6. Export labeled masks to `data/output/`

## Viewing the Web Interface

Open `src/index.html` in a web browser to view the room selector interface.
