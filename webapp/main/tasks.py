# tasks.py
import datetime
from celery import shared_task
from .firebase_helpers import get_camera_by_uuid, save_camera_data
import time
from firebase_admin import db
from .utils import expand_polygon, initialize_tracking, initialize_counting_regions, track_and_count
from ultralytics import YOLO
import cv2

def test_camera_connection(ip_url):
    cap = cv2.VideoCapture(ip_url)
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

@shared_task
def run_detection_for_configured_cameras():
    ambientes_ref = db.reference('/ambientes')
    ambientes_data = ambientes_ref.get()
    if not ambientes_data:
        return

    current_time = datetime.datetime.utcnow().timestamp()

    for ambiente_nombre, ambiente_data in ambientes_data.items():
        for piso_nombre, piso_data in (ambiente_data.get('piso', {}) or {}).items():
            camaras = (piso_data.get('camara', {}) or {})
            for cam_uuid, cam_data in camaras.items():
                if cam_data.get('status') == 'configurated':
                    # Solo detectar si stable
                    if cam_data.get('connection_state', 'stable') == 'stable':
                        detection_interval = cam_data.get('detection_interval', 5)
                        last_detection_time = cam_data.get('last_detection_time', 0)
                        # Verificamos si ya pasó el intervalo
                        if current_time - last_detection_time < detection_interval:
                            # Aún no es tiempo de detectar esta cam
                            continue
                        
                        ip_url = cam_data['ip_url']
                        pixels_per_meter = cam_data['px_m']
                        polygons = cam_data.get('mesas', [])

                        # Intentar leer un frame
                        cap = cv2.VideoCapture(ip_url)
                        if not cap.isOpened():
                            # No se pudo abrir el stream
                            continue
                        ret, frame = cap.read()
                        cap.release()
                        if not ret or frame is None:
                            continue

                        frame = cv2.resize(frame, (640, 480))
                        model = YOLO('yolov8n.pt')

                        # Preparar tracking_data
                        tracking_data_list = []
                        for polygon in polygons:
                            expanded_polygon = expand_polygon(polygon['points'], 0, pixels_per_meter)
                            track_history = initialize_tracking()
                            counting_regions = initialize_counting_regions(expanded_polygon, polygon['max_personas'])
                            # Solo un frame, no trackeo en historico, pero se puede mantener track_history
                            # para el siguiente ciclo si se guarda en Firebase (opcional)
                            frame2 = frame.copy()
                            frame2 = track_and_count(frame2, model, track_history, counting_regions)
                            personas_detectadas = counting_regions[0]['counts']
                            polygon['personas_Detectadas'] = personas_detectadas
                            counting_regions[0]['counts'] = 0
                            tracking_data_list.append({
                                'polygon': polygon
                            })

                        # Guardar resultados
                        cam_data['mesas'] = [td['polygon'] for td in tracking_data_list]
                        cam_data['last_detection_time'] = current_time
                        save_camera_data(ambiente_nombre, piso_nombre, cam_uuid, cam_data)


@shared_task
def start_detection_task(camera_uuid):
    camera_data = get_camera_by_uuid(camera_uuid)
    if not camera_data:
        return

    ip_url = camera_data['ip_url']
    pixels_per_meter = camera_data['px_m']
    detection_interval = camera_data.get('detection_interval', 5)
    polygons = camera_data.get('mesas', [])
    ambiente_nombre = camera_data['ambiente']
    piso = camera_data['piso']

    model = YOLO('yolov8n.pt')
    tracking_data_list = []
    for polygon in polygons:
        expanded_polygon = expand_polygon(polygon['points'], 0, pixels_per_meter)
        track_history = initialize_tracking()
        counting_regions = initialize_counting_regions(expanded_polygon, polygon['max_personas'])
        tracking_data_list.append({
            'polygon': polygon,
            'expanded_polygon': expanded_polygon,
            'track_history': track_history,
            'counting_regions': counting_regions
        })

    while True:
        if not test_camera_connection(ip_url):
            time.sleep(detection_interval)
            continue

        cap = cv2.VideoCapture(ip_url)
        if not cap.isOpened():
            time.sleep(detection_interval)
            continue

        ret, frame = cap.read()
        if not ret:
            cap.release()
            time.sleep(detection_interval)
            continue

        frame = cv2.resize(frame, (640, 480))
        for tracking_data in tracking_data_list:
            frame = track_and_count(
                frame, model, tracking_data['track_history'], tracking_data['counting_regions']
            )
            personas_detectadas = tracking_data['counting_regions'][0]['counts']
            tracking_data['polygon']['personas_Detectadas'] = personas_detectadas
            tracking_data['counting_regions'][0]['counts'] = 0

        camera_data['mesas'] = [td['polygon'] for td in tracking_data_list]
        save_camera_data(ambiente_nombre, piso, camera_uuid, camera_data)
        cap.release()
        time.sleep(detection_interval)


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
                    ip_url = cam_data['ip_url']

                    if connection_state == 'stable':
                        if not test_camera_connection(ip_url):
                            cam_data['connection_state'] = 'reconnecting'
                            cam_data['connection_attempts'] = 1
                            save_camera_data(ambiente_nombre, piso_nombre, cam_uuid, cam_data)
                    elif connection_state == 'reconnecting':
                        if not test_camera_connection(ip_url):
                            connection_attempts += 1
                            cam_data['connection_attempts'] = connection_attempts
                            if connection_attempts >= 5:
                                cam_data['connection_state'] = 'failed'
                            save_camera_data(ambiente_nombre, piso_nombre, cam_uuid, cam_data)
                        else:
                            # Reconexión exitosa
                            cam_data['connection_state'] = 'stable'
                            cam_data['connection_attempts'] = 0
                            save_camera_data(ambiente_nombre, piso_nombre, cam_uuid, cam_data)
                    elif connection_state == 'failed':
                        # Intento indefinido, si se recupera sola:
                        if test_camera_connection(ip_url):
                            cam_data['connection_state'] = 'stable'
                            cam_data['connection_attempts'] = 0
                            save_camera_data(ambiente_nombre, piso_nombre, cam_uuid, cam_data)
