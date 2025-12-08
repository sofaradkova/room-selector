# Self Assessment

## Core ML Fundamentals

- **Modular code design with reusable functions and classes** (3 pts)

  - Organized code into separate modules: `ocr_labels.py`, `matcher.py`, `add_unmatched_masks.py`, `utils.py`
  - Main orchestration in `src/run_pipeline.py` chains these modules together

## Data Collection, Preprocessing, & Feature Engineering

- **Properly normalized or standardized input features/data** (3 pts)

  - `src/ocr_labels.py`:

    - Lines 60-70: Normalize confidence scores to 0-1 range in `ocr_image_tesseract()`
    - Lines 126-145: EasyOCR confidence normalization in `ocr_image_easyocr()`
    - Lines 190-227: Deduplication and normalization in `combine_ocr_results()`

  - `src/matcher.py` (lines 7-18): Centroid normalization for spatial matching

- **Implemented preprocessing pipeline handling data quality issues** (5 pts)

  1. **CLAHE Enhancement** - `src/ocr_labels.py` lines 16-20:
     Applied in `ocr_image_tesseract()` line 38 and `ocr_image_easyocr()` lines 130-132

  2. **Text Normalization** - `src/ocr_labels.py` lines 48-53:

  - Pattern filtering with DEFAULT_PATTERN (line 13)
  - Confidence thresholding (lines 59, 128)

  3. **Duplicate Handling** - `src/ocr_labels.py` lines 190-227:

  - IoU-based deduplication (lines 154-180)
  - Confidence weighting (lines 190-227)

- **Applied feature engineering** (5 pts)

  1. **Centroid Computation** - `src/ocr_labels.py` lines 181-183

     - Derived (cx, cy) coordinates from bounding box data
     - Used as primary spatial feature for label-to-mask matching

  2. **Distance Metric** - `src/ocr_labels.py` lines 186-188

     - Created `_dist()` function for Euclidean distance between centroids
     - Enables spatial proximity matching

  3. **Rectangular Mask Feature** - `src/run_pipeline.py` lines 24-44
     - Transformed irregular SAM segmentation masks into axis-aligned rectangular representations
     - Created derived mask feature with bounded geometry

- **Performed feature selection or dimensionality reduction with justification** (5 pts)

  - `src/matcher.py` lines 48-130: Selected features for matching:
    - Centroid location (lines 62-65)
    - Bbox containment (lines 75-80)
    - Confidence scoring (lines 99-125)

## Model Training & Optimization

- **Trained model using GPU/CUDA acceleration** (5 pts)

  - Ran SAM for automatic mask generation at the first step in Colab using CUDA
    - `automatic_mask_generator.ipynb` - Setup section
  - Ran SAM predictor for unmatched labels using MPS locally
    - `add_unmatched_masks.py` lines 50-56

## Computer Vision

- **Used or fine-tuned vision transformer architecture** (7 pts)

  - Used SAM 2.1 Hiera Large (`src/sam_masks.py` lines 20-35: Model loading from checkpoint)

- **Applied image segmentation** (10 pts)

  - SAM 2.1 model generates instance segmentation masks for individual rooms (`src/sam_masks.py` lines 36-83)
  - Additonal mask generation for unmatched labels using SAM predictor (`src/add_unmatched_masks.py` lines 58+)

## Advanced System Integration

- **Built multi-stage ML pipeline connecting outputs of one model to inputs of another** (7 pts)

  - Stage 1: SAM 2.1 generates initial room masks
  - Stage 2: Dual OCR (Tesseract + EasyOCR) detects room labels
  - Stage 3: Custom matcher associates labels with masks based on spatial features
  - Stage 4: Unmatched labels detected by OCR are passed to SAM 2.1 to generate additional masks
  - Stage 5: Post-processing rectangularizes and exports results
  - `src/run_pipeline.py` orchestrates this process

- **Implemented ensemble method combining predictions from distinct models** (7 pts)

  Dual OCR engines: `src/ocr_labels.py`:

  - Tesseract: `ocr_image_tesseract()` (lines 23-95)
  - EasyOCR: `ocr_image_easyocr()` (lines 96-151)
  - Combination: `combine_ocr_results()` (lines 190-227)

  - Combination logic (lines 190-227):
    - IoU-based deduplication (lines 154-180)
    - Confidence weighting (lines 192-227)
    - Centroid distance matching (lines 186-189)

- **Deployed model as functional web application with user interface** (10 pts)

  - Interactive web interface in `src/index.html`
  - Deployed to GitHub Pages: https://sofaradkova.github.io/room-selector/

## Model Evaluation & Analysis

- **Performed error analysis with visualization or discussion of failure cases** (5 pts)

  - Visualized masks produced by SAM to assess which masks do not correspond to rooms and which are of poor quality. Tested different parameters for SAM, floorplans with text and no text on them, and then multiple filtering options to only output masks corresponding to rooms instead of all spaces. (`automatic_mask_generator.ipynb` - see visualizations in the readme)
  - Tracked unmatched labels from OCR and passed them to SAM for mask regeneration (`add_unmatched_masks.py`)
  - Visualized OCR labels with matched masks, as well as unmatched labels to check for correspondance (`ocr.ipynb` - "Masks with matched labels visualization" section)

- **Compared multiple model architectures or approaches quantitatively** (5 pts)

  - Compared Tesseract vs EasyOCR performance, as well as their combination
  - `ocr.ipynb` - "Tesseract and EasyOCR comparison" section

## Solo Project Credit

- **Completed project individually without a partner** (10 pts)

## Other Contributions Not Captured Above

- **Applied two OCR models for text detection in images** (7 pts)

  - `src/ocr_labels.py` lines 23-151:

    - Tesseract implementation (lines 23-95)
    - EasyOCR implementation (lines 96-151)

    - `notebooks/ocr.ipynb`:
    - Complete OCR exploration and comparison

## Following Directions

- Ontime submission by 5pm on Tuesday, December 9th (note that late submissions will be accepted but only for the normal 72 hour late period, and will not qualify for this rubric item). (3)
- Self-assessment submitted that follows guidelines for at most 15 selections in Machine Learning with evidence (3)
- SETUP.md exists with clear, step-by-step installation instructions (2)
- ATTRIBUTION.md exists with detailed attributions of all sources including AI-generation information (2)
- requirements.txt or environment.yml file is included and accurate (2)
- README.md has What it Does section that describes in one paragraph what your project does (1)
- README.md has Quick Start section that concisely explains how to run your project (1)
- README.md has Video Links section with direct links to your demo and technical walkthrough videos (1)
- README.md has Evaluation section that presents any quantitative results, accuracy metrics, or qualitative outcomes from testing (1)
- README.md has Individual Contributions section for group projects that describes who did what. (1)
- Demo video is of the correct length and appropriate for non-specialist audience with no code shown (2)
- Technical walkthrough is of the correct length and clearly explains code structure, ML techniques, and key contributions (2)
- Attended 1-2 project workshop days (1)
- Attended 3-4 project workshop days (1)

## **Project Cohesion**

- README clearly articulates a single, unified project goal or research question
- Project demo video effectively communicates why the project matters to a non-technical audience in non-technical terms
- Project addresses a real-world problem or explores a meaningful research question
- Technical walkthrough demonstrates how components work together synergistically (not just isolated experiments)
- Project shows clear progression from problem → approach → solution → evaluation
- Design choices are explicitly justified in videos or documentation
- Evaluation metrics directly measure the stated project objectives
- None of the major components awarded rubric item credit in the machine learning category are superfluous to the larger goals of the project (no unrelated "point collecting")
- Clean codebase with readable code and no extraneous, stale, or unused files
