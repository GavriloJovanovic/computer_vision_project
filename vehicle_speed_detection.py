import cv2
from ultralytics import YOLO

# Load the video file
video_path = "car_video.mp4"
cap = cv2.VideoCapture(video_path)

# Load YOLOv8 model
yolo_model = YOLO("yolov8n.pt")


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
        for box in result.boxes.xyxy:
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    cv2.imshow("Vehicle Detection", frame)

    # Press 'q' to exit
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
