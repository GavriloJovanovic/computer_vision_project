# main.py

from processing import detect_vehicles
import cv2

import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="Vehicle Speed Analysis "
                                                 "using YOLO and OpenCV.")
    parser.add_argument("--video", type=str, required=True,
                        help="Path to the input video file.")
    parser.add_argument("--model", type=str, default="yolov8n.pt",
                        help="Path to the YOLO model file.")
    parser.add_argument("--output", type=str, default="vehicle_speeds.json",
                        help="Path to the output JSON file.")
    return parser.parse_args()


def save_fastest_vehicle_image(tracker):
    """Save an image of the fastest vehicle detected."""
    fastest_vehicle_id = tracker.get_fastest_vehicle()
    fastest_vehicle_frame = tracker.get_fastest_vehicle_frame(fastest_vehicle_id)

    if fastest_vehicle_id and fastest_vehicle_frame is not None:
        x1, y1, x2, y2 = tracker.vehicle_last_box[fastest_vehicle_id]
        fastest_vehicle_img = fastest_vehicle_frame[y1:y2, x1:x2]
        filename = f"fastest_vehicle_{fastest_vehicle_id}.png"
        cv2.imwrite(filename, fastest_vehicle_img)
        print(f"Fastest vehicle image saved as {filename}")

        return fastest_vehicle_id  # Vraćamo ID za dalji ispis
    return None


if __name__ == "__main__":
    print("Starting vehicle speed detection...")

    args = parse_args()

    # Start detection of vehicles on the given video
    tracker = detect_vehicles(args.video, args.model, args.output)

    # Wait for the image of the fastest vehicle
    fastest_vehicle_id = save_fastest_vehicle_image(tracker)

    # Print on terminal information about fastest vehicle
    if fastest_vehicle_id:
        vehicle_stats = tracker.vehicle_stats[fastest_vehicle_id]
        min_speed = vehicle_stats["min_speed"] \
            if vehicle_stats["min_speed"] != float("inf") else 0
        avg_speed = vehicle_stats["sum_speed"] / vehicle_stats["count"] \
            if vehicle_stats["count"] > 0 else 0

        print(f"Fastest Vehicle Info: ID={fastest_vehicle_id}, "
              f"Time={tracker.vehicle_start_time[fastest_vehicle_id]:.2f}s, "
              f"Max Speed="
              f"{vehicle_stats['max_speed']:.2f} px/s, "
              f"Min Speed={min_speed:.2f} px/s, "
              f"Avg Speed={avg_speed:.2f} px/s")
    else:
        print("\nNo fastest vehicle detected.")

    print("Vehicle speed data saved as vehicle_speeds.json")
