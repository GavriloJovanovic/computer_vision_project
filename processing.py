# processing.py

import cv2
from tracking import VehicleTracker
from utils import save_json
from config import VIDEO_PATH, MODEL_PATH, OUTPUT_JSON


def detect_vehicles():
    """Detect and track vehicles in the video."""
    cap = cv2.VideoCapture(VIDEO_PATH)
    fps = cap.get(cv2.CAP_PROP_FPS)

    tracker = VehicleTracker(VIDEO_PATH, MODEL_PATH, fps)

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        current_time = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000
        tracker.process_frame(frame, current_time)

        cv2.imshow("Motor Vehicle Detection", frame)
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    # Save vehicle speed data
    save_json(tracker.vehicle_stats, OUTPUT_JSON)

    return tracker
