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

### Qualitative Outcomes

## Contributions

This project was completed individualy by me and AI assistants.
