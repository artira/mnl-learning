# Vampire Survivors — Reward Dynamics Extraction Pipeline

**Media Neuroscience Lab, UC Santa Barbara**
**Dr. René Weber, Principal Investigator**

Computational pipeline for extracting reward event data from Vampire Survivors gameplay videos. Automates detection of level-ups, deaths, and results screens to support the lab's research on gaming reward dynamics and compulsive media use.

## Overview

This pipeline processes recorded Vampire Survivors gameplay sessions and extracts temporally precise event data that would otherwise require frame-by-frame human coding. The automated extraction enables scalable analysis across multiple gameplay sessions for reward trajectory construction and fMRI regressor generation.

## Quick Start

```bash
# 1. Clone the repository
git clone git@github.com:artira/mnl-learning.git
cd mnl-learning/module4_capstone/pipeline

# 2. Set up the environment (one-time)
chmod +x setup.sh
./setup.sh

# 3. Activate environment
conda activate vs-pipeline

# 4. Process a video
python run_pipeline.py /path/to/gameplay_video.mp4
```

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
# Change sample rate (default: 4 fps)
python run_pipeline.py video.mp4 --sample-rate 2

# Custom output directory
python run_pipeline.py video.mp4 --output-dir /path/to/results
```

## Detected Events

| Event | Detection Method | Description |
|---|---|---|
| Level-Up | OCR ("Level Up!" text) | Player reaches a new level, upgrade menu appears |
| Lucky Level-Up | OCR ("Lucky" + "Level Up!") | Bonus level-up with extra choices |
| Death | OCR ("Game Over" / "Death") | Player character dies |
| Results Screen | OCR ("Results" text) | End-of-session results display |

## Output Structure

Each video produces its own output directory:

```
pipeline_results/
  video_name/
    events/
      all_features_ocr.csv         # Per-frame OCR detection results
      detected_levelups.csv        # Grouped level-up events with timestamps
      detected_deaths.csv          # Death events
      detected_results.csv         # Results screen events
      levelup_detection_ocr.png    # 4-panel detection visualization
      event_summary.json           # Summary with event counts and schedule analysis
  batch_summary.json               # Batch processing results (multi-video runs)
```

## Algorithm

The pipeline uses EasyOCR (CRAFT text detection + CRNN text recognition) to read on-screen text from sampled video frames.

For each sampled frame:
1. Crop the center region where event text appears (10-55% height, 25-80% width)
2. Run OCR to detect and read all text in the region
3. Match detected text against event patterns ("Level Up!", "Results", "Game Over")
4. Handle split detections where OCR reads "Level" and "Up !" as separate results
5. Group consecutive detection frames into discrete events with start/end timestamps
6. Merge close events (within 2 seconds) to handle OCR frame-to-frame inconsistency

Sample rate (default 4 fps) controls the tradeoff between speed and detection coverage. Level-up menus typically last 2-10 seconds, so 4 fps provides 8-40 frames per event.

## Validation

Tested against human-coded ground truth for Video3_MultiCharacter.mp4 (9:31 duration, 25 human-coded level-ups):

```
Confusion Matrix:
                    Human: YES    Human: NO
Pipeline: YES         20             0
Pipeline: NO           5            90

Precision:  100.0%  (zero false positives)
Recall:      80.0%  (detected 20 of 25 level-ups)
F1 Score:    0.89
Accuracy:   95.7%
```

The 5 missed level-ups occur when the menu appears for under 1 second (below the sampling window) or visual effects obscure the text on the specific sampled frames. All 20 detections are confirmed correct — the pipeline does not produce false alarms.

## Pipeline Files

| File | Purpose |
|---|---|
| `run_pipeline.py` | Main entry point — handles CLI arguments, video discovery, batch processing |
| `levelup_detector_arti_ocr_v5.py` | Core detection engine — OCR extraction, event grouping, schedule analysis, visualization |
| `environment.yml` | Conda environment specification |
| `setup.sh` | One-command setup script |

## Requirements

- Python 3.10
- OpenCV (video frame reading)
- EasyOCR (text detection and recognition)
- NumPy, Pandas, Matplotlib
- ~500MB disk for EasyOCR models (downloaded automatically on first run)

## Research Context

This pipeline supports the Media Neuroscience Lab's Gaming/Reward Dynamics project, which investigates how reward structures in video games relate to compulsive gameplay patterns. The extracted event timestamps serve as:

- **Reward trajectory data**: Level-up timing patterns classified by Skinner's reinforcement schedules (Fixed Ratio vs Variable Ratio, assessed via coefficient of variation of inter-level-up intervals)
- **fMRI regressors**: Precise event timestamps for modeling neural responses to reward events during in-scanner gameplay sessions
- **Behavioral markers**: Decision time during level-up menus as a measure of player engagement and familiarity with upgrade options

## Author

Arti Ramanathan — M.A. Emerging Media Studies, Boston University (2025)
External Collaborator, Media Neuroscience Lab, UCSB
