from ultralytics import YOLO
import numpy as np
import cv2
from collections import defaultdict
from shapely.geometry import Polygon
from shapely.geometry.point import Point

def initialize_tracking():
    return defaultdict(list)

def initialize_counting_regions(polygon, max_persons):
    return [
        {
            "name": "Table Region",
            "polygon": Polygon(polygon),
            "counts": 0,
            "max_persons": max_persons,
            "region_color": (0, 255, 0),
            "text_color": (255, 255, 255),
        }
    ]

def track_and_count(frame, model, track_history, counting_regions):
    results = model.track(frame, persist=True, classes=[0], conf=0.1, iou=0.5)

    if results[0].boxes.id is not None:
        boxes = results[0].boxes.xyxy.cpu()
        track_ids = results[0].boxes.id.int().cpu().tolist()
        clss = results[0].boxes.cls.cpu().tolist()

        for box, track_id, cls in zip(boxes, track_ids, clss):
            bbox_center = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2

            track = track_history[track_id]
            track.append((float(bbox_center[0]), float(bbox_center[1])))
            if len(track) > 30:
                track.pop(0)
            points = np.hstack(track).astype(np.int32).reshape((-1, 1, 2))
            cv2.polylines(frame, [points], isClosed=False, color=(0, 255, 0), thickness=2)

# Check if detection inside region
            for region in counting_regions:
                if region["polygon"].contains(Point((bbox_center[0], bbox_center[1]))):
                    region["counts"] += 1

    # Draw regions and display counts
    for region in counting_regions:
        region_label = f"{region['counts']}/{region['max_persons']}"
        region_color = region["region_color"]
        region_text_color = region["text_color"]

        polygon_coords = np.array(region["polygon"].exterior.coords, dtype=np.int32)
        centroid_x, centroid_y = int(region["polygon"].centroid.x), int(region["polygon"].centroid.y)

        text_size, _ = cv2.getTextSize(region_label, cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.7, thickness=2)
        text_x = centroid_x - text_size[0] // 2
        text_y = centroid_y + text_size[1] // 2
        cv2.rectangle(frame, (text_x - 5, text_y - text_size[1] - 5), (text_x + text_size[0] + 5, text_y + 5), region_color, -1)
        cv2.putText(frame, region_label, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, region_text_color, 2)
        cv2.polylines(frame, [polygon_coords], isClosed=True, color=region_color, thickness=2)

    # Reset counts for each region
    for region in counting_regions:
        region["counts"] = 0

    return frame

# Load YOLO model
model = YOLO('yolov8s-seg.pt')