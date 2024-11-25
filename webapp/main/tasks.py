from celery import shared_task
from .firebase_helpers import get_camera_by_uuid, save_camera_data
import cv2
import time
import numpy as np
from ultralytics import YOLO
from shapely.geometry import Point
from .utils import expand_polygon, initialize_tracking, initialize_counting_regions, track_and_count

@shared_task
def start_detection_task(camera_uuid):
    camera_data = get_camera_by_uuid(camera_uuid)
    if not camera_data:
        return

    ip_url = camera_data['ip_url']
    pixels_per_meter = camera_data['px_m']
    detection_interval = camera_data.get('detection_interval', 5)  # Default to 5 seconds
    polygons = camera_data.get('mesas', [])

    cap = cv2.VideoCapture(ip_url)
    model = YOLO('yolov8n.pt')

    tracking_data_list = []
    for polygon in polygons:
        expanded_polygon = expand_polygon(polygon['points'], 0.3, pixels_per_meter)
        track_history = initialize_tracking()
        counting_regions = initialize_counting_regions(expanded_polygon, polygon['max_personas'])
        tracking_data_list.append({
            'polygon': polygon,
            'expanded_polygon': expanded_polygon,
            'track_history': track_history,
            'counting_regions': counting_regions
        })

    while True:
        success, frame = cap.read()
        if not success:
            # Retry connection
            cap.release()
            time.sleep(5)  # Wait before retrying
            cap = cv2.VideoCapture(ip_url)
            continue

        frame = cv2.resize(frame, (640, 480))

        # Perform tracking and counting for each polygon
        for tracking_data in tracking_data_list:
            frame = track_and_count(
                frame, model, tracking_data['track_history'], tracking_data['counting_regions']
            )

            # Update personas_Detectadas
            personas_detectadas = tracking_data['counting_regions'][0]['counts']
            tracking_data['polygon']['personas_Detectadas'] = personas_detectadas
            tracking_data['counting_regions'][0]['counts'] = 0  # Reset counts

        # Save updated counts to Firebase
        camera_data['mesas'] = [td['polygon'] for td in tracking_data_list]
        ambiente_nombre = camera_data['ambiente']
        piso = camera_data['piso']
        save_camera_data(ambiente_nombre, piso, camera_uuid, camera_data)

        time.sleep(detection_interval)  # Wait before next detection cycle

    cap.release()
