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


@shared_task
def run_detection_for_configured_cameras():
    ambientes_ref = db.reference('/ambientes')
    ambientes_data = ambientes_ref.get()
    if not ambientes_data:
        return

    current_time = datetime.now(pytz.utc).timestamp()

    # Cargar el modelo una sola vez si es posible (fuera del bucle de cámaras si todas usan el mismo)
    # O dentro del bucle si diferentes cámaras pueden usar diferentes modelos (como está ahora)
    # model = YOLO('yolo11s.pt') # Podría ir aquí si todas usan yolo11s.pt

    for ambiente_nombre, ambiente_data in ambientes_data.items():
        for piso_nombre, piso_data in (ambiente_data.get('piso', {}) or {}).items():
            camaras = (piso_data.get('camara', {}) or {})
            for cam_uuid, cam_data in camaras.items():
                if cam_data.get('status') == 'configurated':
                    if cam_data.get('connection_state', 'stable') == 'stable':
                        detection_interval = cam_data.get('detection_interval', 5)
                        last_detection_time = cam_data.get('last_detection_time', 0)
                        if current_time - last_detection_time < detection_interval:
                            continue

                        ip_url = cam_data['ip_url']
                        username = cam_data.get('username')
                        password = cam_data.get('password')
                        protocol = cam_data.get('protocol', 'http')
                        full_url = construct_full_url(ip_url, username, password, protocol)

                        pixels_per_meter = cam_data['px_m']
                        polygons_data = cam_data.get('mesas', [])
                        if not polygons_data: # Si no hay polígonos, saltar esta cámara
                            continue

                        cap = cv2.VideoCapture(full_url)
                        if not cap.isOpened():
                            # Loggear error de apertura si es necesario
                            continue
                        ret, frame = cap.read()
                        cap.release()
                        if not ret or frame is None:
                            # Loggear error de lectura si es necesario
                            continue

                        # --- Optimización Principal ---
                        # 1. Redimensionar frame una vez
                        frame_resized = cv2.resize(frame, (640, 480))

                        # 2. Cargar modelo (aquí si es específico por cámara, o fuera si es global)
                        model = YOLO('yolo11s.pt') # Asumiendo yolo11s.pt para esta cámara

                        # 3. Ejecutar detección/seguimiento UNA SOLA VEZ en el frame redimensionado
                        #    Usamos track para obtener IDs consistentes entre frames si persist=True
                        results = model.track(frame_resized, persist=True, classes=[0], conf=0.1, iou=0.5, verbose=False) # verbose=False para evitar prints de YOLO

                        # 4. Preparar las regiones de conteo (geometrías)
                        counting_regions = []
                        for idx, p_data in enumerate(polygons_data):
                            try:
                                # Asegúrate que expand_polygon devuelve una lista de tuplas (x, y)
                                expanded_points = expand_polygon(p_data['points'], 0, pixels_per_meter)
                                # Crear objeto Polygon de Shapely para chequeo eficiente
                                polygon_geom = Polygon(expanded_points)
                                counting_regions.append({
                                    'id': p_data.get('id', f'poly_{idx}'), # Usar ID si existe, sino un ID temporal
                                    'polygon_geom': polygon_geom,
                                    'max_personas': p_data.get('max_personas', 0),
                                    'count': 0 # Inicializar contador
                                })
                            except Exception as e:
                                print(f"Error processing polygon {p_data.get('id', idx)} for camera {cam_uuid}: {e}")
                                # Consider logging this error properly
                                continue # Saltar este polígono si hay error

                        # 5. Iterar sobre los resultados de la detección UNA VEZ
                        detections_for_historic = []
                        if results and results[0].boxes.id is not None:
                            boxes = results[0].boxes.xyxy.cpu()
                            track_ids = results[0].boxes.id.int().cpu().tolist()
                            # clss = results[0].boxes.cls.cpu().tolist() # Clases si las necesitas

                            for box, track_id in zip(boxes, track_ids):
                                # Calcular el punto central (o base) de la detección
                                # Usamos el centro del bounding box como referencia
                                bbox_center_x = (box[0] + box[2]) / 2
                                bbox_center_y = (box[1] + box[3]) / 2 # Podrías usar box[3] (base del box) si es más apropiado
                                detection_point = Point(bbox_center_x, bbox_center_y)

                                # Verificar en qué región cae el punto
                                for region in counting_regions:
                                    if region['polygon_geom'].contains(detection_point):
                                        region['count'] += 1
                                        # Si una persona solo puede estar en una región a la vez, puedes usar 'break'
                                        # break

                        # 6. Actualizar los datos originales de los polígonos y preparar histórico
                        updated_polygons_data = []
                        for p_data in polygons_data:
                            p_id = p_data.get('id', None)
                            # Encontrar la región correspondiente por ID (o índice si se usó como fallback)
                            found_region = None
                            if p_id is not None:
                                found_region = next((r for r in counting_regions if r['id'] == p_id), None)
                            # Si no se encontró por ID (o no había ID), podríamos intentar por índice si lo guardamos
                            # Para simplificar, asumimos que el ID existe o el orden se mantiene
                            # Si no, necesitaríamos una forma más robusta de mapear p_data con region

                            if found_region:
                                personas_detectadas = found_region['count']
                            else:
                                # Si por alguna razón no se encontró la región (ej. error previo), asignar 0
                                personas_detectadas = 0
                                print(f"Advertencia: No se encontró la región procesada para el polígono con ID {p_id} en cámara {cam_uuid}")


                            # Actualizar el diccionario original del polígono
                            p_data['personas_Detectadas'] = personas_detectadas
                            updated_polygons_data.append(p_data) # Añadir a la lista actualizada

                            # Añadir al histórico
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

                        # 7. Guardar los datos actualizados
                        cam_data['mesas'] = updated_polygons_data # Guardar la lista con conteos actualizados
                        cam_data['last_detection_time'] = current_time

                        save_camera_data(ambiente_nombre, piso_nombre, cam_uuid, cam_data)

                        # Guardar el histórico en MongoDB
                        if detections_for_historic: # Solo guardar si hubo algo que detectar/contar
                           guardar_historico(cam_uuid, detections_for_historic, ambiente_nombre, piso_nombre)

                        # Limpiar memoria si es necesario (depende de cómo YOLO maneje los recursos)
                        # del model
                        # torch.cuda.empty_cache() # Si usas GPU con PyTorch

                    # else: connection_state != 'stable' -> no hacer nada
                # else: status != 'configurated' -> no hacer nada

    # Nota: La lógica de `track_history` y el dibujo de las trazas y conteos en el frame
    # que estaba en `track_and_count` no está presente en esta versión optimizada,
    # ya que el objetivo principal era optimizar el *conteo* y el guardado de datos.
    # Si necesitas generar un frame con las visualizaciones, tendrías que añadir esa lógica
    # después del paso 5, iterando sobre `results` y `counting_regions` para dibujar.



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