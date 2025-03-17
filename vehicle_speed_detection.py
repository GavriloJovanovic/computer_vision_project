import cv2
from ultralytics import YOLO

# Load the video file
video_path = "car_video.mp4"
cap = cv2.VideoCapture(video_path)

# Load YOLOv8 model
yolo_model = YOLO("yolov8n.pt")

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

    for result in results:
        for box, cls in zip(result.boxes.xyxy, result.boxes.cls):
            class_id = int(cls)  # Convert to integer
            if class_id in MOTOR_VEHICLES_DICT.keys():  # Filter only motor vehicles
                x1, y1, x2, y2 = map(int, box)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"Vehicle: {MOTOR_VEHICLES_DICT[class_id]}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    cv2.imshow("Motor Vehicle Detection", frame)

    # Press 'q' to exit
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
