# Vampire Survivors — Reward Dynamics Pipeline

**Media Neuroscience Lab, UCSB**

Computational pipeline for extracting reward event data from Vampire Survivors gameplay videos. Automates detection of level-ups, gem collection, deaths, and other game events for the lab's reward dynamics research.

## Quick Start

```bash
# 1. Clone or download this folder
# 2. Set up the environment (one-time)
chmod +x setup.sh
./setup.sh

# 3. Activate environment
conda activate vs-pipeline

# 4. Run on your video
python run_pipeline.py your_gameplay_video.mp4
```

## What It Does

The pipeline processes gameplay videos and extracts:

| Feature | Method | Speed | Accuracy (Video3) |
|---|---|---|---|
| Level-Up events | OCR text detection | ~20 min/video | F1=0.89 (20/25 detected) |
| Death events | OCR text detection | (same pass) | — |
| Results screens | OCR text detection | (same pass) | — |
| Blue/Green/Red gems | HSV color detection | ~2 min/video | Needs validation |
| Gem collection events | Frame differencing | (same pass) | Needs validation |

## Usage

### Single video
```bash
python run_pipeline.py gameplay.mp4
```

### Multiple videos
```bash
python run_pipeline.py video1.mp4 video2.mp4 video3.mp4
```

### All videos in a folder
```bash
python run_pipeline.py /path/to/videos/*.mp4
```

### Options
```bash
# Only detect events (faster — skips gem counting)
python run_pipeline.py video.mp4 --events-only

# Only count gems (faster — skips OCR)
python run_pipeline.py video.mp4 --gems-only

# Lower sample rate (faster but may miss short events)
python run_pipeline.py video.mp4 --sample-rate 1

# Custom output directory
python run_pipeline.py video.mp4 --output-dir /path/to/results

# Save debug frames for gem color tuning
python run_pipeline.py video.mp4 --debug
```

## Output Structure

```
pipeline_results/
  video_name/
    events/
      all_features_ocr.csv         # Per-frame OCR results
      detected_levelups.csv        # Grouped level-up events
      detected_deaths.csv          # Death events
      detected_results.csv         # Results screen events
      levelup_detection_ocr.png    # Detection visualization
      event_summary.json           # Summary statistics
    gems/
      gem_counts_per_frame.csv     # Per-frame gem counts
      gem_collection_events.csv    # Collection events
      gem_analysis.png             # Gem visualization
      gem_summary.json             # Summary statistics
      debug_gems/                  # Debug frames (if --debug)
  batch_summary.json               # Overall batch results
```

## Pipeline Files

| File | Algorithm | Purpose |
|---|---|---|
| `run_pipeline.py` | — | Main entry point, handles CLI and batching |
| `levelup_detector_arti_ocr_v5.py` | EasyOCR (CRAFT + CRNN) | Detects level-ups, deaths, results via text |
| `gem_counter_arti.py` | HSV color masking + contours | Counts blue, green, red gems |
| `reward_aggregator.py` | Pandas merge + arithmetic | Combines outputs, computes XP metrics |
| `xp_growth_metrics.py` | Wiki XP formulas | XP requirements and Growth calculations |

## Validation

Tested against human-coded data for Video3_MultiCharacter.mp4:

```
Level-Up Detection Confusion Matrix:
                    Human: YES    Human: NO
Pipeline: YES         20             0
Pipeline: NO           5            90

Precision: 100.0%  (zero false positives)
Recall:    80.0%   (caught 20 of 25 level-ups)
F1 Score:  0.89
Accuracy:  95.7%
```

The 5 missed level-ups occur when the menu appears for under 1 second
(below the 2fps sampling window) or visual effects obscure the text.

## Requirements

- Python 3.10
- OpenCV
- EasyOCR
- NumPy, Pandas, Matplotlib
- ~500MB disk for EasyOCR models (downloaded on first run)

All dependencies are installed via `setup.sh` or `environment.yml`.

## XP Formula Reference

From the Vampire Survivors wiki:

- Levels 1-20: XP requirement increases by +10 per level (5, 15, 25, ...)
- Level 20→21: +600 XP spike
- Levels 21-40: XP requirement increases by +13 per level
- Level 40→41: +2400 XP spike
- Levels 41+: XP requirement increases by +16 per level

Gem XP values (from lab spreadsheet):
- Blue gem: 1.25 XP average
- Green gem: 9 XP
- Red gem: Needed_XP / 3 (scales with level)

## Authors

Arti Ramanathan — Pipeline development
Media Neuroscience Lab — Dr. René Weber, UCSB
