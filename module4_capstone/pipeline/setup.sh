#!/bin/bash
#
# Vampire Survivors Reward Dynamics Pipeline — Setup
# Media Neuroscience Lab, UCSB
#
# Run this once to set up the environment:
#   chmod +x setup.sh
#   ./setup.sh
#

echo "=========================================="
echo "VS Reward Dynamics Pipeline — Setup"
echo "=========================================="

# Check if conda is installed
if ! command -v conda &> /dev/null; then
    echo "ERROR: conda not found."
    echo "Install Anaconda or Miniconda first:"
    echo "  https://docs.conda.io/en/latest/miniconda.html"
    exit 1
fi

echo "[1/3] Creating conda environment..."
conda env create -f environment.yml -y

echo ""
echo "[2/3] Activating environment..."
eval "$(conda shell.bash hook)"
conda activate vs-pipeline

echo ""
echo "[3/3] Verifying installation..."
python3 -c "
import cv2
import numpy
import pandas
import easyocr
import matplotlib
print('  OpenCV:', cv2.__version__)
print('  NumPy:', numpy.__version__)
print('  Pandas:', pandas.__version__)
print('  EasyOCR: OK')
print('  Matplotlib:', matplotlib.__version__)
print()
print('All dependencies installed successfully!')
"

echo ""
echo "=========================================="
echo "Setup complete!"
echo "=========================================="
echo ""
echo "To use the pipeline:"
echo "  conda activate vs-pipeline"
echo "  python run_pipeline.py your_video.mp4"
echo ""
echo "For help:"
echo "  python run_pipeline.py --help"
echo ""
echo "Examples:"
echo "  python run_pipeline.py video1.mp4"
echo "  python run_pipeline.py video1.mp4 video2.mp4"
echo "  python run_pipeline.py *.mp4"
echo "  python run_pipeline.py video1.mp4 --events-only"
echo "  python run_pipeline.py video1.mp4 --sample-rate 1"
