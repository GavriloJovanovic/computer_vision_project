# utils.py

import json


def save_json(data, filename):
    """Save dictionary data to a JSON file."""
    with open(filename, "w") as json_file:
        json.dump(data, json_file, indent=4)


def estimate_speed(track_history, fps):
    """Calculate vehicle speed based on movement in pixels per second."""
    if len(track_history) < 2:
        return 0

    x1, y1 = track_history[-2]
    x2, y2 = track_history[-1]
    distance = ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5
    return distance * fps / 2
