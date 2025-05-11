# PATHvision
# Advanced Vehicle Perception System

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.10+-red.svg)](https://pytorch.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.5+-green.svg)](https://opencv.org/)

A comprehensive system for vehicle perception, combining object detection, tracking, and collision prediction to assist autonomous vehicles in navigating complex environments.

## Features

- **Object Detection & Tracking**: Uses YOLOv9 and ByteTrack for accurate real-time object detection and tracking
- **Road/Walkable Area Detection**: Implements color segmentation and optional Detectron2 integration
- **Trajectory Analysis**: Advanced analysis of object movement patterns
- **Collision Prediction**: Time-to-collision calculation with dynamic risk assessment
- **Action Determination**: Automated decision making (Proceed, Slow Down, Stop)
- **Visualization System**: Comprehensive visual feedback with trajectory paths and collision warnings

![System Demo](assets/fig7.png)
![collision_case1](assets/collision.png)
![collision_case2](assets/collision_2.png)

## Requirements

- Python 3.8+
- PyTorch 1.10+ 
- OpenCV 4.5+
- Ultralytics YOLO
- (Optional) Detectron2 for improved road segmentation

## Installation

1. Clone this repository:

```bash
git clone https://github.com/Bhavisha-06/PATHvision.git
cd PATHvision
```

2. Create and activate a virtual environment:

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install required packages:

```bash
pip install -r requirements.txt
```

4. (Optional) Install Detectron2 for improved road segmentation:

```bash
# Follow the official Detectron2 installation instructions:
# https://detectron2.readthedocs.io/en/latest/tutorials/install.html
```

## Usage

### Basic usage

```bash
python src/main.py --input path/to/video.mp4 --output path/to/output.mp4
```

### Advanced options

```bash
python src/main.py --input path/to/video.mp4 --output path/to/output.mp4 --no-tracking --frame-skip 10
```

### Available Arguments

| Argument | Description |
|----------|-------------|
| `--input` | Input video file path |
| `--output` | Output video file path (will be created) |
| `--no-tracking` | Disable object tracking (faster but less accurate) |
| `--frame-skip` | Process road detection every N frames (higher value = faster processing) |
| `--model` | Path to custom YOLO model (defaults to YOLOv9s) |
| `--conf-thresh` | Detection confidence threshold (default: 0.25) |
| `--use-detectron` | Force use of Detectron2 for road segmentation if available |
| `--device` | Processing device ('cpu', 'cuda', '0', '1', etc.) |

## Project Structure

```
PATHvision/
├── src/                        # Source code
│   ├── __init__.py
│   ├── main.py                 # Main entry point
│   ├── perception/             # Core perception modules
│   │   ├── __init__.py
│   │   ├── detector.py         # Object detection module
│   │   ├── tracker.py          # Object tracking module
│   │   ├── road_detection.py   # Road/walkable area detection
│   │   └── collision.py        # Collision prediction
│   ├── utils/                  # Utility functions
│   │   ├── __init__.py
│   │   ├── visualization.py    # Visualization tools
│   │   └── config.py           # Configuration management
│   └── config/                 # Configuration files
│       ├── default_config.yaml
│       └── bytetrack.yaml      # ByteTrack configuration
├── data/                       # Example data folder
│   └── output/                 # Default output folder
├── docs/                       # Documentation
│   └── images/                 # Images for documentation
├── requirements.txt            # Project dependencies
├── setup.py                    # Package setup file
└── README.md                   # Project readme
```

## Configuration

You can modify the `src/config/default_config.yaml` file to change various settings:

```yaml
# Example configuration 
detection:
  model: yolov9s.pt
  confidence_threshold: 0.25
  
tracking:
  enabled: true
  history_size: 30
  
collision:
  ttc_threshold: 3.0
  danger_threshold: 0.6
  
road_detection:
  use_detectron: auto  # auto, force, disable
  frame_skip: 5
  
visualization:
  show_trajectories: true
  show_collision_paths: true
```

## Examples

### Processing a video file

```python
from pathvision import VehiclePerceptionSystem

# Initialize the system
system = VehiclePerceptionSystem(use_tracking=True)

# Process a video file
system.process_video(
    input_path="path/to/video.mp4",
    output_path="path/to/output.mp4",
    process_every_n_frames=5
)
```

### Real-time processing from a camera

```python
from pathvision import VehiclePerceptionSystem
import cv2

# Initialize the system
system = VehiclePerceptionSystem(use_tracking=True)

# Open webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Process frame
    processed_frame = system.process_frame(frame)
    
    # Display result
    cv2.imshow("PATHvision", processed_frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgements

- [Ultralytics YOLO](https://github.com/ultralytics/ultralytics) for the object detection model
- [ByteTrack](https://github.com/ifzhang/ByteTrack) for the object tracking algorithm
- [Detectron2](https://github.com/facebookresearch/detectron2) for the optional road segmentation
