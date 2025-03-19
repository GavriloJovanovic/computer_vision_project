import cv2
from ultralytics import YOLO
import numpy as np
import json

# Load the video file
video_path = "car_video.mp4"
cap = cv2.VideoCapture(video_path)
fps = cap.get(cv2.CAP_PROP_FPS)

# Load YOLOv8 model
yolo_model = YOLO("yolov8n.pt")


# Function for calculating the speed of the vehicle
def estimate_speed(track_history, fps):
    if len(track_history) < 2:
        return 0

    x1, y1 = track_history[-2]
    x2, y2 = track_history[-1]
    distance = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
    return distance * fps / 2


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

# Vehicle data storage
vehicle_speeds = {}
vehicle_histories = {}
vehicle_stats = {}  # Stores min, max, and average speeds
vehicle_start_time = {}  # Tracks the time when a vehicle appears
vehicle_best_frame = {}  # Tracks the best frame for each vehicle
vehicle_last_box = {}  # Stores last known bounding box of each vehicle
vehicle_max_width = {}  # Stores max width for each vehicle

frame_count = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame_count += 1
    current_time = frame_count / fps

    # Perform object detection (ByteTrack)
    results = yolo_model.track(frame, persist=True, tracker="bytetrack.yaml")

    for result in results:
        for box, cls, conf, track_id in zip(result.boxes.xyxy,
                                            result.boxes.cls,
                                            result.boxes.conf,
                                            result.boxes.id):
            class_id = int(cls)  # Convert to integer
            if class_id in MOTOR_VEHICLES_DICT.keys():
                x1, y1, x2, y2 = map(int, box)
                track_id = int(track_id)

                # Check the width of the vehicle
                vehicle_width = x2 - x1

                # Store the center of the vehicle
                center = ((x1 + x2) // 2, (y1 + y2) // 2)
                if track_id not in vehicle_histories:
                    vehicle_histories[track_id] = []
                    vehicle_stats[track_id] = {'min_speed': float('inf'),
                                               'max_speed': 0,
                                               'sum_speed': 0,
                                               'count': 0}
                    vehicle_start_time[track_id] = current_time
                    vehicle_best_frame[track_id] = frame.copy()
                    vehicle_last_box[track_id] = (x1, y1, x2, y2)
                    vehicle_max_width[track_id] = vehicle_width

                vehicle_histories[track_id].append(center)

                # If the current width is larger than the last stored width,
                # update the best frame
                if vehicle_width > vehicle_max_width[track_id]:
                    vehicle_best_frame[track_id] = frame.copy()
                    # Update max width
                    vehicle_max_width[track_id] = vehicle_width

                # Speed calculation
                speed = estimate_speed(vehicle_histories[track_id], fps)
                vehicle_speeds[track_id] = speed

                # Update vehicle statistics
                if speed > 0:
                    vehicle_stats[track_id]['min_speed'] = \
                        min(vehicle_stats[track_id]['min_speed'], speed)

                vehicle_stats[track_id]['max_speed'] = \
                    max(vehicle_stats[track_id]['max_speed'], speed)
                vehicle_stats[track_id]['sum_speed'] += speed
                vehicle_stats[track_id]['count'] += 1

                # Blue Box for tracking
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                cv2.putText(frame,
                            f"ID {track_id} | {int(speed)} px/s",
                            (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    cv2.imshow("Motor Vehicle Detection", frame)

    # Press 'q' to exit
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# Determine the fastest vehicle based on avg speed in its last frame
fastest_vehicle_id = None
max_global_avg_speed = 0

for track_id, stats in vehicle_stats.items():
    avg_speed = stats['sum_speed'] / stats['count'] \
        if stats['count'] > 0 else 0
    min_speed = stats['min_speed'] \
        if stats['min_speed'] != float('inf') else 0

    # Select the vehicle with the highest average speed
    if avg_speed > max_global_avg_speed:
        max_global_avg_speed = avg_speed
        fastest_vehicle_id = track_id

# Save JSON file with vehicle stats
vehicle_data = []
for track_id, stats in vehicle_stats.items():
    avg_speed = stats['sum_speed'] / stats['count'] \
        if stats['count'] > 0 else 0
    min_speed = stats['min_speed'] if stats['min_speed'] != float('inf') else 0
    vehicle_data.append({
        "ID": track_id,
        "time": vehicle_start_time[track_id],
        "max_speed": stats['max_speed'],
        "min_speed": min_speed,
        "avg_speed": avg_speed
    })

with open("vehicle_speeds.json", "w") as json_file:
    json.dump(vehicle_data, json_file, indent=4)

# Save the fastest vehicle's segmented image
if fastest_vehicle_id and fastest_vehicle_id in vehicle_best_frame:
    fastest_vehicle_frame = vehicle_best_frame[fastest_vehicle_id]
    x1, y1, x2, y2 = vehicle_last_box[fastest_vehicle_id]
    fastest_vehicle_img = fastest_vehicle_frame[y1:y2, x1:x2]
    filename = f"fastest_vehicle_{fastest_vehicle_id}.png"
    cv2.imwrite(filename, fastest_vehicle_img)
    print(f"Segmented image of the fastest vehicle saved as {filename}")

    print(f"Fastest Vehicle Info: ID={fastest_vehicle_id}, "
          f"Time={vehicle_start_time[fastest_vehicle_id]:.2f}s, "
          f"Max Speed="
          f"{vehicle_stats[fastest_vehicle_id]['max_speed']:.2f} px/s, "
          f"Min Speed={min_speed:.2f} px/s, "
          f"Avg Speed={max_global_avg_speed:.2f} px/s")

print("Vehicle speed data saved as vehicle_speeds.json")
