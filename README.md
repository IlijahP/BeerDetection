# Beer Detection with YOLO

This project performs live beer detection using a YOLO model through your webcam.

## Files

- `detect_beer.py` - Live webcam detection script with buffering
- `yolo_beer_detection_tutorial.ipynb` - Educational notebook for training and understanding YOLO
- `pyproject.toml` - Project dependencies
- `best.pt` - Your trained YOLO model (place here)

## Quick Start

1. Place your YOLO model file (`best.pt`) in the project root directory.
2. Install dependencies: `uv sync`
3. Run the detection: `python detect_beer.py`

## Educational Notebook

For a complete learning experience, open `yolo_beer_detection_tutorial.ipynb` which covers:
- YOLO fundamentals
- Data preparation
- Model training
- Evaluation
- Inference and deployment

## Requirements

- Python 3.8+
- Webcam
- YOLO model trained on alcohol-related objects

## Usage

Run the script and point your webcam at beer bottles, glasses, etc. The model will detect and highlight:
- alchol-bottle
- beer-bottle
- beer-glass
- shot
- wine-glass

Press 'q' to quit the detection window.

## Model

Make sure your `best.pt` model is compatible with Ultralytics YOLO26 format.
