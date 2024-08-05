from ultralytics import YOLO, solutions
import numpy as np
import cv2
from collections import defaultdict
from shapely.geometry import Polygon
from shapely.geometry.point import Point

# Camera calibration parameters
camera_matrix = np.array([
    [1.42981228e+03, 0.00000000e+00, 6.79469267e+02],
    [0.00000000e+00, 1.43227361e+03, 3.52205086e+02],
    [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]
], dtype=np.float32)
dist_coeffs = np.array([0, 0, 0, 0, 0], dtype=np.float32)

# Load an official or custom model
model = YOLO('yolov8s-seg.pt')

# Open the webcam
cap = cv2.VideoCapture(0)
assert cap.isOpened(), "Error reading video file"
width, height, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))

print(f"Dimensiones del stream: {width}x{height}")

# Tracking history
track_history = defaultdict(list)

# Counting regions
counting_regions = [
    {
        "name": "YOLOv8 Polygon Region",
        "polygon": Polygon([(50, 80), (250, 20), (450, 80), (400, 350), (100, 350)]),
        "counts": 0,
        "dragging": False,
        "region_color": (255, 42, 4),
        "text_color": (255, 255, 255),
    },
    {
        "name": "YOLOv8 Rectangle Region",
        "polygon": Polygon([(200, 250), (440, 250), (440, 550), (200, 550)]),
        "counts": 0,
        "dragging": False,
        "region_color": (37, 255, 225),
        "text_color": (0, 0, 0),
    },
]

current_region = None

def mouse_callback(event, x, y, flags, param):
    global current_region

    if event == cv2.EVENT_LBUTTONDOWN:
        for region in counting_regions:
            if region["polygon"].contains(Point((x, y))):
                current_region = region
                current_region["dragging"] = True
                current_region["offset_x"] = x
                current_region["offset_y"] = y

    elif event == cv2.EVENT_MOUSEMOVE:
        if current_region is not None and current_region["dragging"]:
            dx = x - current_region["offset_x"]
            dy = y - current_region["offset_y"]
            current_region["polygon"] = Polygon(
                [(p[0] + dx, p[1] + dy) for p in current_region["polygon"].exterior.coords]
            )
            current_region["offset_x"] = x
            current_region["offset_y"] = y

    elif event == cv2.EVENT_LBUTTONUP:
        if current_region is not None and current_region["dragging"]:
            current_region["dragging"] = False

cv2.namedWindow("YOLOv8 Tracking")
cv2.setMouseCallback("YOLOv8 Tracking", mouse_callback)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Run YOLOv8 tracking on the frame, persisting tracks between frames
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

    # Draw regions
    for region in counting_regions:
        region_label = str(region["counts"])
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

    # Visualize the results on the frame
    annotated_frame = results[0].plot()

    # Display the annotated frame
    cv2.imshow("YOLOv8 Tracking", annotated_frame)

    # Reset counts for each region
    for region in counting_regions:
        region["counts"] = 0

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close all windows
cap.release()
cv2.destroyAllWindows()