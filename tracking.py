# tracking.py

import cv2
from ultralytics import YOLO
from config import MOTOR_VEHICLES_DICT
from utils import estimate_speed


class VehicleTracker:
    """Class to track vehicle speeds and determine the fastest vehicle."""

    def __init__(self, video_path, model_path, fps):
        self.cap = cv2.VideoCapture(video_path)
        self.fps = fps
        self.yolo_model = YOLO(model_path)

        # Vehicle tracking data
        self.vehicle_histories = {}
        self.vehicle_stats = {}  # Stores min, max, and average speeds
        self.vehicle_start_time = {}   # Tracks the time when a vehicle appears
        self.vehicle_best_frame = {}  # Tracks the best frame for each vehicle
        self.vehicle_last_box = {}  # Stores last known bounding box of each vehicle
        self.vehicle_best_confidence = {}  # Stores the best confidence score

    def process_frame(self, frame, current_time):
        """Process a single video frame and track vehicles."""
        results = self.yolo_model.track(frame, persist=True, tracker="bytetrack.yaml")

        for result in results:
            for box, cls, conf, track_id in zip(result.boxes.xyxy,
                                                result.boxes.cls,
                                                result.boxes.conf,
                                                result.boxes.id):
                class_id = int(cls)  # Convert to integer
                if class_id not in MOTOR_VEHICLES_DICT:
                    continue

                x1, y1, x2, y2 = map(int, box)
                track_id = int(track_id)
                # Store the center of the vehicle
                center = ((x1 + x2) // 2, (y1 + y2) // 2)

                confidence = float(conf)  # Extract confidence score

                # Initialize tracking data
                if track_id not in self.vehicle_histories:
                    self.vehicle_histories[track_id] = []
                    self.vehicle_stats[track_id] = {'min_speed': float('inf'),
                                                    'max_speed': 0,
                                                    'sum_speed': 0,
                                                    'count': 0}
                    self.vehicle_start_time[track_id] = current_time
                    self.vehicle_best_frame[track_id] = frame.copy()
                    self.vehicle_last_box[track_id] = (x1, y1, x2, y2)
                    # Initialize confidence score
                    self.vehicle_best_confidence[track_id] = confidence

                self.vehicle_histories[track_id].append(center)

                # Update the best frame based on confidence score
                if confidence > self.vehicle_best_confidence[track_id]:
                    self.vehicle_best_frame[track_id] = frame.copy()
                    self.vehicle_last_box[track_id] = (x1, y1, x2, y2)
                    # Update best confidence score
                    self.vehicle_best_confidence[track_id] = confidence

                # Speed calculation
                speed = estimate_speed(self.vehicle_histories[track_id], self.fps)

                # Update vehicle statistics
                if speed > 0:
                    self.vehicle_stats[track_id]['min_speed'] = \
                        min(self.vehicle_stats[track_id]['min_speed'], speed)

                self.vehicle_stats[track_id]['max_speed'] = \
                    max(self.vehicle_stats[track_id]['max_speed'], speed)
                self.vehicle_stats[track_id]['sum_speed'] += speed
                self.vehicle_stats[track_id]['count'] += 1

                # Blue Box for tracking
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                cv2.putText(frame, f"ID {track_id} | {int(speed)} px/s"
                                   f" | {confidence:.2f}",
                            (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    def get_fastest_vehicle(self):
        """Find the fastest vehicle based on average speed."""
        fastest_vehicle_id = None
        max_global_avg_speed = 0

        for track_id, stats in self.vehicle_stats.items():
            avg_speed = stats['sum_speed'] / stats['count'] if stats['count'] > 0 else 0
            if avg_speed > max_global_avg_speed:
                max_global_avg_speed = avg_speed
                fastest_vehicle_id = track_id

        return fastest_vehicle_id

    def get_fastest_vehicle_frame(self, fastest_vehicle_id):
        """Return the best (widest) frame of the fastest vehicle."""
        return self.vehicle_best_frame.get(fastest_vehicle_id, None)
