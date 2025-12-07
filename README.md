# Duke Housing Room Selection

## What It Does

This project is meant to help Duke students pick rooms during the housing selection process. Right now when entering the housing portal we can only see the room numbers and have to constantly cross-reference them with floor plans published on a separate webpage and manually search for the right room. The choice has to be done under the constraint of a 5-minute selection time slot, so the project is meant to address this inconvenience. This is an MVP for one floor plan, but in the future, it will be extended to view floor plans for all Duke quads.

## Quick Start

1. **Install dependencies**

   ```bash
   conda env create -f environment.yml
   conda activate room-selector
   ```

2. **Download SAM 2.1 checkpoint**

   - Download `sam2.1_hiera_large.pt` from [Meta's SAM 2.1 releases](https://github.com/facebookresearch/sam2)
   - Place in `checkpoints/` directory

3. **Run the pipeline**

   ```bash
   python src/run_pipeline.py
   ```

4. **View the interface**
   - Open `src/index.html` in a web browser
   - Or visit: https://sofaradkova.github.io/room-selector/

See [SETUP.md](SETUP.md) for detailed installation instructions.

## Video Links

- **Demo Video**: [INSERT DEMO VIDEO LINK]
- **Technical Walkthrough**: [INSERT TECHNICAL WALKTHROUGH LINK]

## Evaluation

### Quantitative Results

#### OCR Models Comparison (Tesseract VS EasyOCR)

| floorplan_name | tesseract_identified | tesseract_filtered_out | easyocr_identified | easyocr_filtered_out | combined_labels | easyocr_contributed |
| -------------- | -------------------- | ---------------------- | ------------------ | -------------------- | --------------- | ------------------- |
| floorplan1     | 2                    | 34                     | 0                  | 24                   | 2               | 0                   |
| floorplan2     | 16                   | 46                     | 1                  | 36                   | 16              | 0                   |
| floorplan3     | 13                   | 57                     | 4                  | 39                   | 13              | 0                   |
| floorplan4     | 6                    | 57                     | 2                  | 30                   | 7               | 1                   |
| floorplan5     | 16                   | 70                     | 2                  | 43                   | 16              | 0                   |
| floorplan6     | 18                   | 35                     | 11                 | 26                   | 19              | 1                   |

### Qualitative Outcomes

## Contributions

This project was completed individualy by me and AI assistants.
