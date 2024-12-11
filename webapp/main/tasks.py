# tasks.py
from celery import shared_task
from .firebase_helpers import get_camera_by_uuid, save_camera_data
import time
import requests
from firebase_admin import db
from .utils import expand_polygon, initialize_tracking, initialize_counting_regions, track_and_count
from ultralytics import YOLO

@shared_task
def start_detection_task(camera_uuid):
    camera_data = get_camera_by_uuid(camera_uuid)
    if not camera_data:
        return

    ip_url = camera_data['ip_url']
    pixels_per_meter = camera_data['px_m']
    detection_interval = camera_data.get('detection_interval', 5)  # Default 5s
    polygons = camera_data.get('mesas', [])

    # Inicialización del modelo y tracking
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
        # Verificación previa de conexión
        if not test_camera_connection(ip_url):
            # Si la cámara no responde, esperamos y volvemos a intentar luego,
            # pero no lanzamos error, el estado lo maneja la otra tarea.
            time.sleep(detection_interval)
            continue

        # Si la conexión parece estable, abrimos el stream
        cap = cv2.VideoCapture(ip_url)
        if not cap.isOpened():
            time.sleep(detection_interval)
            continue

        success, frame = cap.read()
        if not success:
            cap.release()
            time.sleep(detection_interval)
            continue

        frame = cv2.resize(frame, (640, 480))

        # Tracking y conteo
        for tracking_data in tracking_data_list:
            frame = track_and_count(
                frame, model, tracking_data['track_history'], tracking_data['counting_regions']
            )
            personas_detectadas = tracking_data['counting_regions'][0]['counts']
            tracking_data['polygon']['personas_Detectadas'] = personas_detectadas
            tracking_data['counting_regions'][0]['counts'] = 0  # Reset

        # Guardar resultados en Firebase
        camera_data['mesas'] = [td['polygon'] for td in tracking_data_list]
        ambiente_nombre = camera_data['ambiente']
        piso = camera_data['piso']
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
        pisos = ambiente_data.get('piso', {})
        if not isinstance(pisos, dict):
            continue
        for piso_nombre, piso_data in pisos.items():
            camaras = piso_data.get('camara', {})
            if not isinstance(camaras, dict):
                continue
            for cam_uuid, cam_data in camaras.items():
                if cam_data.get('status') == 'configurated':
                    connection_state = cam_data.get('connection_state', 'stable')
                    connection_attempts = cam_data.get('connection_attempts', 0)
                    ip_url = cam_data['ip_url']

                    if connection_state == 'stable':
                        if not test_camera_connection(ip_url):
                            cam_data['connection_state'] = 'reconnecting'
                            cam_data['connection_attempts'] = 1
                            save_camera_data(cam_data['ambiente'], cam_data['piso'], cam_uuid, cam_data)
                    elif connection_state == 'reconnecting':
                        if not test_camera_connection(ip_url):
                            connection_attempts += 1
                            cam_data['connection_attempts'] = connection_attempts
                            if connection_attempts >= 5:
                                cam_data['connection_state'] = 'failed'
                            save_camera_data(cam_data['ambiente'], cam_data['piso'], cam_uuid, cam_data)
                        else:
                            # Se reconectó con éxito
                            cam_data['connection_state'] = 'stable'
                            cam_data['connection_attempts'] = 0
                            save_camera_data(cam_data['ambiente'], cam_data['piso'], cam_uuid, cam_data)
                    elif connection_state == 'failed':
                        # Espera hasta que la cámara se recupere sola
                        # con este modelo, sin intervención del usuario
                        # simplemente dejamos la cámara en estado failed
                        # y la tarea cada minuto intentará nuevamente.
                        if test_camera_connection(ip_url):
                            # Si se recupera sola
                            cam_data['connection_state'] = 'stable'
                            cam_data['connection_attempts'] = 0
                            save_camera_data(cam_data['ambiente'], cam_data['piso'], cam_uuid, cam_data)


def test_camera_connection(ip_url):
    # Hacemos una petición HTTP simple con timeout
    # Si responde 200 OK asumimos que está accesible.
    try:
        r = requests.get(ip_url, timeout=5)
        return r.status_code == 200
    except requests.RequestException:
        return False
