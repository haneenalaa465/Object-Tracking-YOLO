# Object Tracking Comparison with YOLO Models

This repository contains the implementation and results of comparing three YOLO model variants (YOLOv10n, YOLOv11n, and YOLOv12n) for object tracking on video.

## Overview

In this assignment, we evaluate the performance of different lightweight YOLO models on object tracking tasks, with a focus on inference speed. The same video is processed through each model to provide a fair comparison of their performance characteristics.

### YOLOv10n
https://github.com/user-attachments/assets/152619da-99d5-4332-b484-ec8e429eada7



## Project Structure

```
object-tracking-comparison/
├── data/
│   └── video.mp4                          # Input video for tracking
├── models/
│   ├── yolov10n.pt                        # YOLOv10n weights
│   ├── yolov11n.pt                        # YOLOv11n weights
│   └── yolov12n.pt                        # YOLOv12n weights
├── results/
│   ├── comparison_results.json            # Results in JSON format
│   ├── comparison_results.txt             # Tabular results summary
│   ├── yolov10n_output.mp4                # Processed video with YOLOv10n
│   ├── yolov10n_output_metrics.txt        # Performance metrics for YOLOv10n
│   ├── yolov11n_output.mp4                # Processed video with YOLOv11n
│   ├── yolov11n_output_metrics.txt        # Performance metrics for YOLOv11n
│   ├── yolov12n_output.mp4                # Processed video with YOLOv12n
│   └── yolov12n_output_metrics.txt        # Performance metrics for YOLOv12n
├── src/
│   ├── main.py                            # Main processing script
│   ├── run_comparison.py                  # Script to run all models and compare
│   ├── track_yolov10n.py                  # YOLOv10n tracking implementation
│   ├── track_yolov11n.py                  # YOLOv11n tracking implementation
│   ├── track_yolov12n.py                  # YOLOv12n tracking implementation
│   └── utils.py                           # Shared utility functions (optional)
├── report.pdf                             # Detailed report with comparison results
├── requirements.txt                       # Python dependencies
└── README.md                              # This file
```

## Models Used

- **YOLOv10n**: The nano version of YOLOv10
- **YOLOv11n**: The nano version of YOLOv11
- **YOLOv12n**: The nano version of YOLOv12

The "n" (nano) versions of these models are specifically designed for speed and efficiency while maintaining acceptable accuracy, making them suitable for real-time applications or deployment on devices with limited computational resources.

## Installation

1. Clone this repository:
```bash
git clone https://github.com/username/object-tracking-comparison.git
cd object-tracking-comparison
```

2. Set up a Python environment and install dependencies:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

3. Place your input video in the `data/` directory or specify the path when running the scripts.

## Usage

### Running Individual Models

To process the video with a specific model:

```bash
# Run tracking with YOLOv10n
python src/track_yolov10n.py --video data/video.mp4 --output results/yolov10n_output.mp4

# Run tracking with YOLOv11n
python src/track_yolov11n.py --video data/video.mp4 --output results/yolov11n_output.mp4

# Run tracking with YOLOv12n
python src/track_yolov12n.py --video data/video.mp4 --output results/yolov12n_output.mp4
```

### Running All Models for Comparison

To run all models in sequence and generate a comparison report:

```bash
python src/run_comparison.py --video data/video.mp4
```

This will process the video with all three models and save the results and metrics in the `results/` directory.

### Options

- `--video`: Path to the input video file (required)
- `--output`: Path for the output video (default is in results directory)
- `--conf`: Confidence threshold for detection (default: 0.3)
- `--device`: Device to run inference on (cuda/cpu, default: cuda if available)
- `--skip-existing`: Skip models that already have output files (only for run_comparison.py)

## Results and Analysis

After running the comparison, you'll find several output files in the `results/` directory:

1. Processed videos showing object tracking for each model
2. Metrics files containing performance data for each model
3. Comparison results in both text and JSON formats

The main metrics used for comparison are:
- Average FPS (Frames Per Second)
- Total processing time for the entire video
- Model size

### Sample Results

| Model    | Average FPS | Processing Time (s) | Execution Time (s) |
|----------|-------------|---------------------|-----------------|
| YOLOv10n | 4.59        | 205.3               | 234.11          |
| YOLOv11n | 5.32        | 176.95              | 216.8           |
| YOLOv12n | 4.88        | 193.16              | 230.41          |

For a detailed analysis of these results, including performance metrics and visual comparison, please refer to the [full report](report.pdf).


## Requirements

- Python 3.8+
- PyTorch 2.0+
- Ultralytics YOLO package
- OpenCV 4.5+
- Other dependencies listed in requirements.txt

## Hardware Recommendations

For optimal performance when comparing these models, we recommend:
- CUDA-enabled GPU with at least 4GB VRAM
- 8GB+ system RAM
- SSD storage for faster video processing
