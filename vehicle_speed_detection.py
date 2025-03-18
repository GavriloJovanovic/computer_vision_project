import cv2
from ultralytics import YOLO
from sort.sort import Sort
import numpy as np

# Load the video file
video_path = "car_video.mp4"
cap = cv2.VideoCapture(video_path)

# Load YOLOv8 model
yolo_model = YOLO("yolov8n.pt")

# Initialize sort
tracker = Sort()

# COCO class labels relevant to motor vehicles
MOTOR_VEHICLES_DICT = {
    2: "Car",
    3: "Motorcycle",
    5: "Bus",
    7: "Truck"
}

if not cap.isOpened():
    print("Error: Cannot open video file.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Perform object detection
    results = yolo_model(frame)

    detections = []  # Store detected vehicles

    for result in results:
        for box, cls, conf in zip(result.boxes.xyxy,
                                  result.boxes.cls,
                                  result.boxes.conf):
            class_id = int(cls)  # Convert to integer
            if class_id in MOTOR_VEHICLES_DICT.keys():
                x1, y1, x2, y2 = map(int, box)
                confidence = float(conf)
                # Format: x1, y1, x2, y2, score
                detections.append([x1, y1, x2, y2, confidence])

    # Convert detections to numpy array for SORT
    detections = np.array(detections)

    # Update SORT tracker
    tracked_objects = tracker.update(detections)

    # Draw tracked bounding boxes with unique IDs
    for obj in tracked_objects:
        x1, y1, x2, y2, track_id = map(int, obj)
        # Blue box for tracking
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
        cv2.putText(frame, f"ID {track_id}", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    cv2.imshow("Motor Vehicle Detection", frame)

    # Press 'q' to exit
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
