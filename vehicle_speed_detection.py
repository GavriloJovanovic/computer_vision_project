import cv2
from ultralytics import YOLO
import numpy as np

# Load the video file
video_path = "car_video.mp4"
cap = cv2.VideoCapture(video_path)
fps = cap.get(cv2.CAP_PROP_FPS)

# Load YOLOv8 model
yolo_model = YOLO("yolov8n.pt")

# function for calculating the speed of the vehicle
def estimate_speed(track_history, fps):
    if len(track_history) < 2:
        return 0

    x1, y1 = track_history[-2]
    x2, y2 = track_history[-1]
    distance = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
    return distance * fps

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

# Vehicle histories and speeds
vehicle_speeds = {}
vehicle_histories = {}

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Perform object detection (ByteTrack)
    results = yolo_model.track(frame, persist=True, tracker="bytetrack.yaml")

    detections = [] # Store detected vehicles

    for result in results:
        for box, cls, conf, track_id in zip(result.boxes.xyxy,
                                            result.boxes.cls,
                                            result.boxes.conf,
                                            result.boxes.id):
            class_id = int(cls) # Convert to integer
            if class_id in MOTOR_VEHICLES_DICT.keys():
                x1, y1, x2, y2 = map(int, box)
                confidence = float(conf)
                track_id = int(track_id)

                # Store the center of the vehicle
                center = ((x1 + x2) // 2, (y1 + y2) // 2)
                if track_id not in vehicle_histories:
                    vehicle_histories[track_id] = []

                vehicle_histories[track_id].append(center)

                # Speed calculation
                speed = estimate_speed(vehicle_histories[track_id], fps)
                vehicle_speeds[track_id] = speed

                # Blue box for tracking
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                cv2.putText(frame,
                            f"ID {track_id} | {int(vehicle_speeds[track_id])} px/s",
                            (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    cv2.imshow("Motor Vehicle Detection", frame)

    # Press 'q' to exit
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
