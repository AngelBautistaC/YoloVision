# utils.py
import cv2
import numpy as np
import math
from shapely.geometry import Polygon, Point
import collections
from pymongo import MongoClient
from django.conf import settings
from datetime import datetime



def expand_polygon(points, distance_meters, pixels_per_meter):
    distance_pixels = distance_meters * pixels_per_meter
    polygon = Polygon(points)
    expanded_polygon = polygon.buffer(distance_pixels)
    expanded_points = list(expanded_polygon.exterior.coords)
    return expanded_points

def initialize_tracking():
    return collections.defaultdict(list)

def initialize_counting_regions(polygon_points, max_persons):
    polygon = Polygon(polygon_points)
    region_color = (0, 255, 0)
    text_color = (255, 255, 255)
    return [{
        "polygon": polygon,
        "counts": 0,
        "max_persons": max_persons,
        "region_color": region_color,
        "text_color": text_color
    }]

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

    return frame

# -------- MongoDB Histórico de Detecciones -------- #
from pymongo import MongoClient
from django.conf import settings
from datetime import datetime
from firebase_admin import db

def obtener_coleccion_historico():
    cliente = MongoClient(settings.DATABASES['default']['CLIENT']['host'])
    db = cliente['logs_db']
    return db['historico_detecciones']

def guardar_historico(camera_uuid, detections, ambiente, piso):
    historico_detecciones = obtener_coleccion_historico()
    documento = {
        "camera_uuid": camera_uuid,
        "ambiente": ambiente,  # nuevo campo
        "piso": piso,          # nuevo campo
        "timestamp": datetime.utcnow(),
        "detections": detections
    }
    historico_detecciones.insert_one(documento)

