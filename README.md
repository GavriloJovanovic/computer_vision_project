# Computer vision project

This project detects and tracks vehicles in a video, estimates their speed, and identifies the fastest vehicle. It uses OpenCV and YOLO for vehicle detection and tracking.

## Features
- Detect and track vehicles in a traffic video
- Estimate the speed of each vehicle (in pixels per second)
- Identify the fastest vehicle based on its average speed
- Save an image of the fastest vehicle
- Export vehicle speed data to a JSON file

## Prerequisites
Ensure you have Python **3.9+** installed on your system. You will also need `git` to clone the repository.

## Installation Guide

### 1. Clone the repository
```bash
git clone https://github.com/GavriloJovanovic/computer_vision_project.git
cd computer_vision_project
```

### 2. Install dependencies
First, create a virtual environment (optional but recommended):
```bash
python -m venv venv
source venv/bin/activate  # On macOS/Linux
venv\Scripts\activate    # On Windows
```
Then install the required packages:
```bash
pip install -r requirements.txt
```

## Download the Video File
The project does not contain the video file due to storage limitations. You need to download it manually from the following link:
[Download Video](https://drive.google.com/file/d/1GoH1bOYnt8cqBtFbYjFPqqruG1Gncaav/view)

Save the downloaded file as `car_video.mp4` in the project root directory.

## Running the Project

To start the vehicle detection and speed estimation, run:
```bash
python main.py
```

The program will:
1. Load the YOLO model for vehicle detection.
2. Process the video and track vehicles.
3. Calculate their speed in pixels per second.
4. Identify the fastest vehicle and save its image.
5. Store speed data in `vehicle_speeds.json`.

## Output Files
- `vehicle_speeds.json` - JSON file containing speed data of detected vehicles.
- `fastest_vehicle_<ID>.png` - Image of the fastest detected vehicle.

## Project Structure
```
computer_vision_project/
│── config.py            # Configuration file (paths and parameters)
│── main.py              # Main entry point of the project
│── tracking.py          # Handles vehicle detection and tracking
│── processing.py        # Processes video frames and calls the tracker
│── utils.py             # Utility functions (speed estimation, JSON handling)
│── requirements.txt     # Required dependencies
```

## Notes
- The video processing window can be closed by pressing `q`.
- The speed is estimated in **pixels per second**; if real-world speed is needed, a calibration factor must be applied.
- If YOLO does not detect vehicles properly, try using a different YOLO model (`yolov8m.pt`, `yolov8l.pt` etc.).
