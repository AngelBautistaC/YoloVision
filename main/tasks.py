# tasks.py
from datetime import datetime, timedelta, timezone
from celery import shared_task
import pytz
from .firebase_helpers import get_camera_by_uuid, save_camera_data
import time
from firebase_admin import db
from .utils import expand_polygon, guardar_historico, initialize_tracking, initialize_counting_regions, track_and_count
from ultralytics import YOLO
import cv2
import re  # Importar el módulo re
from django.http import HttpRequest
from logger.utils import create_log_entry, create_celery_log_entry
from shapely.geometry import Polygon, Point


def construct_full_url(ip_url, username=None, password=None, protocol="http"):
    """Construye la URL completa para la cámara, incluyendo credenciales y protocolo."""
    if username and password:
        ip_url_no_protocol = re.sub(r"^(https?|rtsp)://", "", ip_url, flags=re.IGNORECASE)
        full_url = f"{protocol.lower()}://{username}:{password}@{ip_url_no_protocol}"
    else:
        full_url = ip_url
        if not (full_url.lower().startswith("http://") or full_url.lower().startswith("https://") or full_url.lower().startswith("rtsp://")):
            full_url = f"{protocol.lower()}://{full_url}"
    return full_url



def test_camera_connection(ip_url, username=None, password=None, protocol="http"):
    """
    Prueba la conexión de la cámara, manejando URLs con y sin credenciales y diferentes protocolos.
    """
    full_url = construct_full_url(ip_url, username, password, protocol)
    cap = cv2.VideoCapture(full_url)
    start = time.time()
    success = False
    while time.time() - start < 5:
        ret, frame = cap.read()
        if ret and frame is not None:
            success = True
            break
        time.sleep(0.5)
    cap.release()
    return success

from django.core.cache import cache


@shared_task
def run_detection_for_configured_cameras():
    ambientes_ref = db.reference('/ambientes')
    ambientes_data = ambientes_ref.get()
    if not ambientes_data:
        return

    current_time = datetime.now(pytz.utc).timestamp()

    for ambiente_nombre, ambiente_data in ambientes_data.items():
        for piso_nombre, piso_data in (ambiente_data.get('piso', {}) or {}).items():
            camaras = (piso_data.get('camara', {}) or {})
            for cam_uuid, cam_data in camaras.items():
                if cam_data.get('status') == 'configurated' and cam_data.get('connection_state', 'stable') == 'stable':
                    detection_interval = cam_data.get('detection_interval', 5)
                    last_detection_time = cam_data.get('last_detection_time', 0)

                    if current_time - last_detection_time < detection_interval:
                        continue

                    lock_key = f"camera_task_active_{cam_uuid}"
                    if cache.get(lock_key):
                        print(f"[IGNORADA] Ya hay tarea activa para {cam_uuid}")
                        continue

                    cache.set(lock_key, True, timeout=60)  # Lock por 60s

                    run_detection_for_camera.delay(ambiente_nombre, piso_nombre, cam_uuid)


@shared_task
def run_detection_for_camera(ambiente_nombre, piso_nombre, cam_uuid):
    lock_key = f"camera_task_active_{cam_uuid}"

    try:
        current_time = datetime.now(pytz.utc).timestamp()

        cam_ref = db.reference(f'/ambientes/{ambiente_nombre}/piso/{piso_nombre}/camara/{cam_uuid}')
        cam_data = cam_ref.get()
        if not cam_data or cam_data.get('status') != 'configurated' or cam_data.get('connection_state') != 'stable':
            return

        nombre_camara = cam_data.get('label')
        ip_url = cam_data['ip_url']
        username = cam_data.get('username')
        password = cam_data.get('password')
        protocol = cam_data.get('protocol', 'http')
        full_url = construct_full_url(ip_url, username, password, protocol)

        pixels_per_meter = cam_data['px_m']
        polygons_data = cam_data.get('mesas', [])
        if not polygons_data:
            return

        cap = cv2.VideoCapture(full_url)
        if not cap.isOpened():
            return
        ret, frame = cap.read()
        cap.release()
        if not ret or frame is None:
            return

        frame_resized = cv2.resize(frame, (640, 480))
        model = YOLO('yolo11s.pt')
        results = model.track(frame_resized, persist=True, classes=[0], conf=0.1, iou=0.5, verbose=False)

        counting_regions = []
        for idx, p_data in enumerate(polygons_data):
            try:
                expanded_points = expand_polygon(p_data['points'], 0, pixels_per_meter)
                polygon_geom = Polygon(expanded_points)
                counting_regions.append({
                    'id': p_data.get('id', f'poly_{idx}'),
                    'polygon_geom': polygon_geom,
                    'max_personas': p_data.get('max_personas', 0),
                    'count': 0
                })
            except Exception as e:
                print(f"Error processing polygon {p_data.get('id', idx)} for camera {cam_uuid}: {e}")
                continue

        detections_for_historic = []
        if results and results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu()
            track_ids = results[0].boxes.id.int().cpu().tolist()
            for box, track_id in zip(boxes, track_ids):
                bbox_center_x = (box[0] + box[2]) / 2
                bbox_center_y = (box[1] + box[3]) / 2
                detection_point = Point(bbox_center_x, bbox_center_y)

                for region in counting_regions:
                    if region['polygon_geom'].contains(detection_point):
                        region['count'] += 1

        updated_polygons_data = []
        for p_data in polygons_data:
            p_id = p_data.get('id')
            found_region = next((r for r in counting_regions if r['id'] == p_id), None)
            personas_detectadas = found_region['count'] if found_region else 0
            if not found_region:
                print(f"Advertencia: No se encontró la región procesada para el polígono con ID {p_id} en cámara {cam_uuid}")
            p_data['personas_Detectadas'] = personas_detectadas
            updated_polygons_data.append(p_data)

            detections_for_historic.append({
                'id': p_data.get('id', ''),
                'label': p_data.get('label', ''),
                'personas_detectadas': personas_detectadas,
                'max_personas': p_data.get('max_personas', 0),
                'area': p_data.get('area', 0),
                'perimeter': p_data.get('perimeter', 0),
                'points': p_data.get('points', []),
                'color': p_data.get('color', [0, 255, 0])
            })

        cam_data['mesas'] = updated_polygons_data
        cam_data['last_detection_time'] = current_time
        save_camera_data(ambiente_nombre, piso_nombre, cam_uuid, cam_data)

        if detections_for_historic:
            guardar_historico(cam_uuid, detections_for_historic, ambiente_nombre, piso_nombre, nombre_camara)

    finally:
        # 🔓 Liberar el lock incluso si ocurre un error
        cache.delete(lock_key)




@shared_task
def check_connection_task():
    ambientes_ref = db.reference('/ambientes')
    ambientes_data = ambientes_ref.get()
    if not ambientes_data:
        return

    for ambiente_nombre, ambiente_data in ambientes_data.items():
        for piso_nombre, piso_data in (ambiente_data.get('piso', {}) or {}).items():
            camaras = (piso_data.get('camara', {}) or {})
            for cam_uuid, cam_data in camaras.items():
                if cam_data.get('status') == 'configurated':
                    connection_state = cam_data.get('connection_state', 'stable')
                    connection_attempts = cam_data.get('connection_attempts', 0)
                    ip_url = cam_data.get('ip_url')
                    cam_label = cam_data.get('label', 'N/A')
                    username = cam_data.get('username')  # Obtener usuario
                    password = cam_data.get('password')  # Obtener contraseña
                    protocol = cam_data.get('protocol', 'http')  # Obtener protocolo


                    if connection_state == 'stable':
                        if not test_camera_connection(ip_url, username, password, protocol):  # Pasar credenciales y protocolo
                            cam_data['connection_state'] = 'reconnecting'
                            cam_data['connection_attempts'] = 1
                            save_camera_data(ambiente_nombre, piso_nombre, cam_uuid, cam_data)
                            create_celery_log_entry('Problema de conexión', f'Cámara {cam_label} (UUID: {cam_uuid}, IP: {ip_url}) en {ambiente_nombre}/{piso_nombre} pasó a estado \'reconectando\' tras un intento fallido de conexión .')
                    elif connection_state == 'reconnecting':
                        if not test_camera_connection(ip_url, username, password, protocol):  # Pasar credenciales y protocolo
                            connection_attempts += 1
                            cam_data['connection_attempts'] = connection_attempts
                            if connection_attempts >= 5:
                                cam_data['connection_state'] = 'failed'
                                create_celery_log_entry('Conexión fallida', f'Cámara {cam_label} (UUID: {cam_uuid}, IP: {ip_url}) en {ambiente_nombre}/{piso_nombre} pasó a estado \'failed\' después de 5 minutos sin conexión.')
                            save_camera_data(ambiente_nombre, piso_nombre, cam_uuid, cam_data)
                        else:
                            cam_data['connection_state'] = 'stable'
                            cam_data['connection_attempts'] = 0
                            save_camera_data(ambiente_nombre, piso_nombre, cam_uuid, cam_data)
                            create_celery_log_entry('Conexión recuperada', f'Cámara {cam_label} (UUID: {cam_uuid}, IP: {ip_url}) en {ambiente_nombre}/{piso_nombre} recuperó conexión después de {connection_attempts} minutos.')
                    elif connection_state == 'failed':
                        connection_attempts +=1
                        cam_data['connection_attempts'] = connection_attempts
                        if test_camera_connection(ip_url, username, password, protocol):  # Pasar credenciales y protocolo
                            cam_data['connection_state'] = 'stable'
                            cam_data['connection_attempts'] = 0
                            create_celery_log_entry('Conexión recuperada', f'Cámara {cam_label} (UUID: {cam_uuid}, IP: {ip_url}) en {ambiente_nombre}/{piso_nombre} recuperó conexión después de {connection_attempts} minutos.')
                            save_camera_data(ambiente_nombre, piso_nombre, cam_uuid, cam_data)