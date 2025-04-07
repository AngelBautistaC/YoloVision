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


@shared_task
def run_detection_for_configured_cameras():


    ambientes_ref = db.reference('/ambientes')
    ambientes_data = ambientes_ref.get()
    if not ambientes_data:
        return

    #current_time = datetime.now(pytz.timezone('America/Lima')).timestamp()
    #current_time = datetime.utcnow().timestamp()
    #current_time = datetime.now(timezone.utc).timestamp()
    current_time = datetime.now(pytz.utc).timestamp()

    for ambiente_nombre, ambiente_data in ambientes_data.items():
        for piso_nombre, piso_data in (ambiente_data.get('piso', {}) or {}).items():
            camaras = (piso_data.get('camara', {}) or {})
            for cam_uuid, cam_data in camaras.items():
                if cam_data.get('status') == 'configurated':
                    if cam_data.get('connection_state', 'stable') == 'stable':
                        detection_interval = cam_data.get('detection_interval', 5)
                        last_detection_time = cam_data.get('last_detection_time', 0)
                        if current_time - last_detection_time < detection_interval:
                            #print("No es el momento de detectar")
                            continue
                        #print("es el momento de detectar")

                        ip_url = cam_data['ip_url']
                        username = cam_data.get('username')
                        password = cam_data.get('password')
                        protocol = cam_data.get('protocol', 'http')
                        full_url = construct_full_url(ip_url, username, password, protocol)

                        pixels_per_meter = cam_data['px_m']
                        polygons = cam_data.get('mesas', [])

                        cap = cv2.VideoCapture(full_url)
                        if not cap.isOpened():
                            continue
                        ret, frame = cap.read()
                        cap.release()
                        if not ret or frame is None:
                            continue

                        frame = cv2.resize(frame, (640, 480))
                        model = YOLO('yolo11s.pt')

                        detections_for_historic = []

                        for polygon in polygons:
                            expanded_polygon = expand_polygon(polygon['points'], 0, pixels_per_meter)
                            track_history = initialize_tracking()
                            counting_regions = initialize_counting_regions(expanded_polygon, polygon['max_personas'])

                            frame2 = frame.copy()
                            frame2 = track_and_count(frame2, model, track_history, counting_regions)
                            personas_detectadas = counting_regions[0]['counts']
                            polygon['personas_Detectadas'] = personas_detectadas
                            counting_regions[0]['counts'] = 0

                            detections_for_historic.append({
                                'id': polygon.get('id', ''),
                                'label': polygon.get('label', ''),
                                'personas_detectadas': personas_detectadas,
                                'max_personas': polygon.get('max_personas', 0),
                                'area': polygon.get('area', 0),
                                'perimeter': polygon.get('perimeter', 0),
                                'points': polygon.get('points', []),
                                'color': polygon.get('color', [0, 255, 0])
                            })

                        cam_data['mesas'] = polygons
                        cam_data['last_detection_time'] = current_time

                        #print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
                        #dt = datetime.utcfromtimestamp(current_time)
                        #print(dt.strftime("%Y-%m-%d %H:%M:%S.%f UTC"))
                        #print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")


                        save_camera_data(ambiente_nombre, piso_nombre, cam_uuid, cam_data)

                        # Aquí guardamos el histórico en MongoDB
                        guardar_historico(cam_uuid, detections_for_historic, ambiente_nombre, piso_nombre)





@shared_task
def start_detection_task(camera_uuid):
    camera_data = get_camera_by_uuid(camera_uuid)
    if not camera_data:
        return

    ip_url = camera_data['ip_url']
    username = camera_data.get('username')  # Obtener usuario
    password = camera_data.get('password')  # Obtener contraseña
    protocol = camera_data.get('protocol', 'http')  # Obtener protocolo
    full_url = construct_full_url(ip_url, username, password, protocol)  # Usar la función

    pixels_per_meter = camera_data['px_m']
    detection_interval = camera_data.get('detection_interval', 5)
    polygons = camera_data.get('mesas', [])
    ambiente_nombre = camera_data['ambiente']
    piso = camera_data['piso']

    model = YOLO('yolo11n.pt')
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
        if not test_camera_connection(ip_url, username, password, protocol):  # Pasar credenciales y protocolo
            time.sleep(detection_interval)
            continue

        cap = cv2.VideoCapture(full_url)  # Usar full_url
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