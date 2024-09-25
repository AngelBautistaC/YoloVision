from collections import defaultdict
import time
import uuid
from django.shortcuts import render, redirect
import cv2
import logging
from django.http import HttpResponse, StreamingHttpResponse, HttpResponseBadRequest,JsonResponse
import urllib.request
from django.shortcuts import render
import numpy as np
from shapely.geometry import Polygon
from shapely.geometry.point import Point
from ultralytics import YOLO
import base64
import math
from firebase_admin import db
from .firebase_helpers import get_camera_by_ip, save_camera_data,get_camera_by_uuid,save_camera_data2
from .tasks import start_detection_task
from .utils import expand_polygon, initialize_tracking, initialize_counting_regions, track_and_count


logger = logging.getLogger(__name__)





def configurar_deteccion(request):
    data = load_json_data()
    camaras_calibradas = []
    for ambiente in data['ambientes']:
        for piso, piso_data in ambiente['piso'].items():
            for camara in piso_data.get('camara', []):
                if 'px_m' in camara:
                    camaras_calibradas.append({
                        'ambiente': ambiente['nombreAmbiente'],
                        'piso': piso,
                        'label': camara['label'],
                        'ip_url': camara['ip_url']
                    })
    
    context = {'camaras': camaras_calibradas}
    return render(request, 'configurar_deteccion.html', context)



def iniciar_deteccion(request):
    if request.method == 'POST':
        ip_url = request.POST.get('ip_url')
        data = load_json_data()
        camara_data = None
        for ambiente in data['ambientes']:
            for piso, piso_data in ambiente['piso'].items():
                for camara in piso_data.get('camara', []):
                    if camara['ip_url'] == ip_url:
                        camara_data = camara
                        break
                if camara_data:
                    break
            if camara_data:
                break
        
        if not camara_data:
            return JsonResponse({'error': 'Cámara no encontrada'})
        
        # Guardar datos de la cámara en la sesión
        request.session['camera_data'] = camara_data
        
        # Set tracking_active to False
        request.session['tracking_active'] = False
        
        return JsonResponse({'success': True})
    
    return JsonResponse({'error': 'Método no permitido'})


def obtener_detecciones(request):
    detected_polygons = request.session.get('detections', [])
    manual_polygons = request.session.get('manual_polygons', [])
    all_polygons = detected_polygons + manual_polygons
    return JsonResponse({'detections': all_polygons})

def detectar_mesas(request):
    if request.method == 'POST':
        camera_data = request.session.get('camera_data')
        if not camera_data:
            return JsonResponse({'error': 'No hay datos de cámara en la sesión'})

        ip_url = camera_data['ip_url']
        pixels_per_meter = camera_data['px_m']

        cap = cv2.VideoCapture(ip_url)
        success, frame = cap.read()
        cap.release()

        if not success:
            return JsonResponse({'error': 'No se pudo capturar el frame'})

        # Resize frame to 640x480
        frame = cv2.resize(frame, (640, 480))

        model = YOLO('yolov8n-seg.pt')

        results = model(frame, classes=[60], conf=0.1)

        detected_polygons = []

        for result in results:
            masks = result.masks
            if masks is not None:
                for i, mask in enumerate(masks):
                    mask_array = mask.xy[0].astype(int)
                    polygon = cv2.convexHull(mask_array)
                    simplified_polygon = simplify_polygon(polygon)
                    if simplified_polygon is not None:
                        simplified_polygon = np.array(simplified_polygon, dtype=np.float32)
                        if len(simplified_polygon) > 2:
                            # Calculate additional information
                            area_pixels = cv2.contourArea(simplified_polygon)
                            area_meters = area_pixels / (pixels_per_meter ** 2)
                            perimeter_pixels = cv2.arcLength(simplified_polygon, True)
                            perimeter_meters = perimeter_pixels / pixels_per_meter
                            personas = math.floor(perimeter_meters / 1.15)

                            detected_polygons.append({
                                'id': str(uuid.uuid4()),
                                'points': simplified_polygon.tolist(),
                                'label': f"Tabla {i+1}",
                                'color': generate_random_color(),
                                'type': 'detected',
                                'area': area_meters,
                                'perimeter': perimeter_meters,
                                'max_personas': personas
                            })
        
        request.session['detections'] = detected_polygons

        return JsonResponse({'success': True, 'detections': detected_polygons})

    return JsonResponse({'error': 'Método no permitido'})


def iniciar_tracking(request):
    if request.method == 'POST':
        data = json.loads(request.body)
        polygons = data.get('polygons', [])
        camera_data = request.session.get('camera_data')
        
        if not camera_data:
            return JsonResponse({'error': 'No hay datos de cámara en la sesión'})

        ip_url = camera_data['ip_url']
        
        # Obtener o crear el UUID de la cámara
        camera_uuid, existing_camera_data = get_camera_by_ip(ip_url)
        if not camera_uuid:
            camera_uuid = str(uuid.uuid4())
            camera_data['uuid'] = camera_uuid
        else:
            camera_data = existing_camera_data

        # Actualizar los datos de la cámara con los polígonos
        camera_data['mesas'] = polygons
        camera_data['detection_interval'] = data.get('detection_interval', 5)  # Intervalo de detección en segundos

        # Usar el ambiente y piso almacenado en la sesión
        ambiente_nombre = request.session.get('ambiente')
        piso = request.session.get('piso')

        # Guardar en Firebase
        save_camera_data2(ambiente_nombre, piso, camera_uuid, camera_data)

        # Iniciar la tarea de detección
        start_detection_task.delay(camera_uuid)

        return JsonResponse({'success': True, 'camera_uuid': camera_uuid})

    return JsonResponse({'error': 'Método no permitido'})





def stream_video(request):
    camera_data = request.session.get('camera_data')
    if not camera_data:
        return JsonResponse({'error': 'No hay datos de cámara en la sesión'})

    ip_url = camera_data['ip_url']
    pixels_per_meter = camera_data['px_m']
    tracking_active = request.session.get('tracking_active', False)

    def generate_frames():
        cap = cv2.VideoCapture(ip_url)

        # Load model based on tracking state
        if tracking_active:
            model = YOLO('yolov8n.pt')  # Model for people detection
        else:
            model = None  # No model needed for just drawing polygons

        # Load polygons from 'database.json' for the current camera
        db_data = load_json_data()
        camera = None
        for ambiente in db_data['ambientes']:
            for piso, piso_data in ambiente['piso'].items():
                for cam in piso_data.get('camara', []):
                    if cam['ip_url'] == ip_url:
                        camera = cam
                        break
            if camera:
                break
        if not camera:
            cap.release()
            return  # or yield an error frame

        polygons = camera.get('mesas', [])

        # Initialize tracking data only if tracking is active
        if tracking_active and model:
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
                break
            frame = cv2.resize(frame, (640, 480))

            if tracking_active and model:
                # Perform tracking and counting for each polygon
                for tracking_data in tracking_data_list:
                    frame = track_and_count(
                        frame, model, tracking_data['track_history'], tracking_data['counting_regions']
                    )

                # Update personas_Detectadas and reset counts
                for tracking_data in tracking_data_list:
                    polygon = tracking_data['polygon']
                    personas_detectadas = tracking_data['counting_regions'][0]['counts']
                    polygon['personas_Detectadas'] = personas_detectadas
                    tracking_data['counting_regions'][0]['counts'] = 0  # Reset counts
            else:
                # No tracking; personas_Detectadas is zero
                for polygon in polygons:
                    polygon['personas_Detectadas'] = 0

            # Draw polygons
            for polygon in polygons:
                points = np.array(polygon['points'], dtype=np.int32)
                cv2.polylines(frame, [points], isClosed=True, color=polygon['color'], thickness=2)
                text_position = tuple(map(int, points[0]))
                cv2.putText(frame, polygon['label'], text_position,
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, polygon['color'], 2)

                # Mostrar información adicional
                info_text = f"Area: {polygon['area']:.2f}m² | Max: {polygon['max_personas']} | Detectadas: {polygon.get('personas_Detectadas', 0)}"
                cv2.putText(frame, info_text, (text_position[0], text_position[1] + 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, polygon['color'], 1)

            _, buffer = cv2.imencode('.jpg', frame)
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

        cap.release()

    return StreamingHttpResponse(generate_frames(), content_type='multipart/x-mixed-replace; boundary=frame')




def generate_random_color():
    return tuple(np.random.randint(0, 255, 3).tolist())


def etiquetar_poligono(request):
    if request.method == 'POST':
        polygon_id = request.POST.get('polygon_id')
        new_label = request.POST.get('label')
        detections = request.session.get('detections', [])
        manual_polygons = request.session.get('manual_polygons', [])
        
        updated = False
        
        for detection in detections + manual_polygons:
            if detection['id'] == polygon_id:
                detection['label'] = new_label
                updated = True
                break

        # Update in database.json if necessary
        camera_data = request.session.get('camera_data')
        if camera_data:
            ip_url = camera_data['ip_url']
            db_data = load_json_data()
            camera = None
            for ambiente in db_data['ambientes']:
                for piso, piso_data in ambiente['piso'].items():
                    for cam in piso_data.get('camara', []):
                        if cam['ip_url'] == ip_url:
                            camera = cam
                            break
            if camera:
                mesas = camera.get('mesas', [])
                for mesa in mesas:
                    if mesa['id'] == polygon_id:
                        mesa['label'] = new_label
                        save_json_data(db_data)
                        break
        
        if updated:
            request.session['detections'] = detections
            request.session['manual_polygons'] = manual_polygons
            return JsonResponse({'success': True})
        
        return JsonResponse({'error': 'Polígono no encontrado'})
    
    return JsonResponse({'error': 'Método no permitido'})




def agregar_poligono(request):
    if request.method == 'POST':
        points = json.loads(request.POST.get('points'))
        camera_data = request.session.get('camera_data')
        pixels_per_meter = camera_data['px_m']
        
        new_polygon = {
            'id': str(uuid.uuid4()),
            'points': points,
            'label': f"Polígono {len(request.session.get('manual_polygons', [])) + 1}",
            'color': generate_random_color(),
            'type': 'manual'
        }
        
        # Calcular información adicional
        area_pixels = cv2.contourArea(np.array(points, dtype=np.int32))
        area_meters = area_pixels / (pixels_per_meter ** 2)
        perimeter_pixels = cv2.arcLength(np.array(points, dtype=np.int32), True)
        perimeter_meters = perimeter_pixels / pixels_per_meter
        personas = math.floor(perimeter_meters / 1.15)  # 1.15 es personas_por_perimetro
        
        new_polygon['area'] = area_meters
        new_polygon['perimeter'] = perimeter_meters
        new_polygon['max_personas'] = personas
        
        manual_polygons = request.session.get('manual_polygons', [])
        manual_polygons = [p for p in manual_polygons if p['id'] != new_polygon['id']]
        manual_polygons.append(new_polygon)
        request.session['manual_polygons'] = manual_polygons

        return JsonResponse({'success': True, 'polygon': new_polygon})
    
    return JsonResponse({'error': 'Método no permitido'})



def quitar_poligono(request):
    if request.method == 'POST':
        polygon_id = request.POST.get('polygon_id')
        detections = request.session.get('detections', [])
        manual_polygons = request.session.get('manual_polygons', [])
        
        detections = [d for d in detections if d['id'] != polygon_id]
        manual_polygons = [p for p in manual_polygons if p['id'] != polygon_id]
        
        request.session['detections'] = detections
        request.session['manual_polygons'] = manual_polygons
        
        return JsonResponse({'success': True})
    
    return JsonResponse({'error': 'Método no permitido'})


def actualizar_poligono(request):
    if request.method == 'POST':
        polygon_id = request.POST.get('id')
        new_points = json.loads(request.POST.get('points'))
        camera_data = request.session.get('camera_data')
        pixels_per_meter = camera_data['px_m']
        
        detected_polygons = request.session.get('detections', [])
        manual_polygons = request.session.get('manual_polygons', [])
        
        for polygon_list in [detected_polygons, manual_polygons]:
            for polygon in polygon_list:
                if polygon['id'] == polygon_id:
                    polygon['points'] = new_points
                    
                    # Recalculate area, perimeter, and max persons
                    area_pixels = cv2.contourArea(np.array(new_points, dtype=np.int32))
                    area_meters = area_pixels / (pixels_per_meter ** 2)
                    perimeter_pixels = cv2.arcLength(np.array(new_points, dtype=np.int32), True)
                    perimeter_meters = perimeter_pixels / pixels_per_meter
                    personas = math.floor(perimeter_meters / 1.15)
                    
                    polygon['area'] = area_meters
                    polygon['perimeter'] = perimeter_meters
                    polygon['max_personas'] = personas
                    
                    request.session['detections'] = detected_polygons
                    request.session['manual_polygons'] = manual_polygons
                    
                    return JsonResponse({
                        'success': True,
                        'polygon': polygon
                    })
        
        return JsonResponse({'error': 'Polígono no encontrado'})
    
    return JsonResponse({'error': 'Método no permitido'})



def simplify_polygon(polygon, target_points=6):
    epsilon = 0.01 * cv2.arcLength(polygon, True)
    approx = cv2.approxPolyDP(polygon, epsilon, True)
    while len(approx) > target_points:
        epsilon *= 1.1
        approx = cv2.approxPolyDP(polygon, epsilon, True)
    if len(approx) < 3:  # Cannot form a polygon with less than 3 points
        return None
    if len(approx) < target_points:
        approx = np.array(approx).squeeze()
        perimeter = cv2.arcLength(approx, True)
        target_distance = perimeter / target_points
        new_points = []
        for i in range(len(approx)):
            p1 = approx[i]
            p2 = approx[(i + 1) % len(approx)]
            distance = np.linalg.norm(p1 - p2)
            if distance > target_distance:
                num_new_points = int(distance / target_distance)
                for j in range(num_new_points):
                    t = (j + 1) / (num_new_points + 1)
                    new_point = (1 - t) * p1 + t * p2
                    new_points.append(new_point)
            new_points.append(p1)
        approx = np.array(new_points)
    return approx.squeeze()





def obtener_info_mesas(request):
    camera_data = request.session.get('camera_data')
    if not camera_data:
        return JsonResponse({'error': 'No hay datos de cámara en la sesión'})

    ip_url = camera_data['ip_url']
    # Load polygons from 'database.json' for the current camera
    db_data = load_json_data()
    camera = None
    for ambiente in db_data['ambientes']:
        for piso, piso_data in ambiente['piso'].items():
            for cam in piso_data.get('camara', []):
                if cam['ip_url'] == ip_url:
                    camera = cam
                    break
    if not camera:
        return JsonResponse({'error': 'Cámara no encontrada en la base de datos'})

    backend_polygons = camera.get('mesas', [])

    # Mark backend polygons with type 'backend'
    for polygon in backend_polygons:
        polygon['type'] = 'backend'

    # Also include polygons from session (detections and manual_polygons)
    detected_polygons = request.session.get('detections', [])
    manual_polygons = request.session.get('manual_polygons', [])

    return JsonResponse({'mesas': detected_polygons + manual_polygons, 'backend_polygons': backend_polygons})



def quitar_poligono(request):
    if request.method == 'POST':
        polygon_id = request.POST.get('polygon_id')
        # Remove from session
        detections = request.session.get('detections', [])
        manual_polygons = request.session.get('manual_polygons', [])
        detections = [d for d in detections if d['id'] != polygon_id]
        manual_polygons = [p for p in manual_polygons if p['id'] != polygon_id]
        request.session['detections'] = detections
        request.session['manual_polygons'] = manual_polygons

        # Remove from database.json
        camera_data = request.session.get('camera_data')
        if camera_data:
            ip_url = camera_data['ip_url']
            db_data = load_json_data()
            camera = None
            for ambiente in db_data['ambientes']:
                for piso, piso_data in ambiente['piso'].items():
                    for cam in piso_data.get('camara', []):
                        if cam['ip_url'] == ip_url:
                            camera = cam
                            break
            if camera:
                mesas = camera.get('mesas', [])
                mesas = [m for m in mesas if m['id'] != polygon_id]
                camera['mesas'] = mesas
                save_json_data(db_data)

        return JsonResponse({'success': True})

    return JsonResponse({'error': 'Método no permitido'})



#----------------------------------------------------------------------------------------------------------------------------------------




####SECCION BASE DE DATOS JSON : CAMBIAR A BASE DE DATOS FIREBASE:
#----------------------------------------------------------------------------------------------------------------------------------------
import json
import os
from django.conf import settings


JSON_FILE_PATH = os.path.join(settings.BASE_DIR, 'data', 'database.json')
#----------------------------------------------------------------------------------------------------------------------------------------

def login_view(request):
    if request.method == 'POST':
        # Aquí iría la lógica de autenticación en el futuro
        return redirect('monitoreo')
    return render(request, 'login.html')

def monitoring_view(request):
    return render(request, 'monitoreo.html')

def configuracion_view(request):
    return render(request, 'configuracion.html')




logger = logging.getLogger(__name__)

def calibrar_camara_paso1(request):
    context = {'step': '1', 'camera_connected': False}

    if request.method == 'POST' and request.headers.get('X-Requested-With') == 'XMLHttpRequest':
        
        ip_port = request.POST.get('ip_port')
        camera_name = request.POST.get('camera_name')
        ambiente = request.POST.get('ambiente')
        piso = request.POST.get('piso', '1')  # Asumimos piso 1 por defecto

        if not ip_port:
            return JsonResponse({'success': False, 'message': 'La URL del stream está vacía'})

        try:
            cap = cv2.VideoCapture(ip_port)
            if not cap.isOpened():
                logger.error(f"No se pudo abrir el stream: {ip_port}")
                return JsonResponse({'success': False, 'message': 'No se pudo abrir el stream de la cámara. Verifica la URL y el formato.'})

            frame_read = False
            for _ in range(30):
                ret, frame = cap.read()
                if ret and frame is not None:
                    frame_read = True
                    break

            cap.release()


            if frame_read:
                request.session['ip_port'] = ip_port
                request.session['camera_name'] = camera_name
                request.session['ambiente'] = ambiente
                request.session['piso'] = piso
                logger.info(f"Conexión exitosa. Datos guardados en sesión.")
                return JsonResponse({'success': True, 'message': 'Conexión exitosa'})

            else:
                logger.error("No se pudo leer un frame después de 30 intentos")
                return JsonResponse({'success': False, 'message': 'No se pudo leer un frame de la cámara después de 30 intentos'})
        except Exception as e:
            logger.exception(f"Error inesperado al conectar con la cámara: {str(e)}")
            return JsonResponse({'success': False, 'message': f'Error inesperado: {str(e)}'})

    return render(request, 'calibrar_camara.html', context)


def calibrar_camara_paso2(request):
    context = {'step': '2'}
    ip_port = request.session.get('ip_port')
    context['ip_port'] = ip_port

    # Definir las dimensiones del tablero de ajedrez
    CHECKERBOARD = (6, 9)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    if request.method == 'POST' and request.headers.get('X-Requested-With') == 'XMLHttpRequest':
        action = request.POST.get('action')
        logger.info(f"Paso 2 - Acción: {action}, URL del stream: {ip_port}")

        if not ip_port:
            logger.error("La URL del stream no está disponible en la sesión")
            return JsonResponse({'success': False, 'message': 'La URL del stream no está disponible. Por favor, vuelve al paso 1.'})

        if action == 'capture':
            try:
                cap = cv2.VideoCapture(ip_port)
                if not cap.isOpened():
                    logger.error(f"No se pudo abrir el stream para captura: {ip_port}")
                    return JsonResponse({'success': False, 'message': 'No se pudo abrir el stream de la cámara.'})

                ret, frame = cap.read()
                cap.release()

                if not ret or frame is None:
                    logger.error("No se pudo capturar la imagen")
                    return JsonResponse({'success': False, 'message': 'No se pudo capturar la imagen'})

                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, 
                                                         cv2.CALIB_CB_ADAPTIVE_THRESH + 
                                                         cv2.CALIB_CB_FAST_CHECK + 
                                                         cv2.CALIB_CB_NORMALIZE_IMAGE)
                if ret:
                    objp = np.zeros((1, CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
                    objp[0,:,:2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
                    
                    corners2 = cv2.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)
                    
                    objpoints = request.session.get('objpoints', [])
                    imgpoints = request.session.get('imgpoints', [])
                    
                    objpoints.append(objp.tolist())  # Convertir a lista para serialización
                    imgpoints.append(corners2.tolist())  # Convertir a lista para serialización
                    
                    request.session['objpoints'] = objpoints
                    request.session['imgpoints'] = imgpoints
                    
                    logger.info(f"Captura exitosa. Total de capturas: {len(objpoints)}")
                    return JsonResponse({'success': True, 'captures': len(objpoints)})
                else:
                    logger.warning("No se detectó el patrón del tablero")
                    return JsonResponse({'success': False, 'message': 'No se detectó el patrón del tablero'})
            except Exception as e:
                logger.exception(f"Error al capturar: {str(e)}")
                return JsonResponse({'success': False, 'message': f'Error al capturar: {str(e)}'})

        elif action == 'calibrate':
            objpoints = request.session.get('objpoints', [])
            imgpoints = request.session.get('imgpoints', [])
            
            if len(objpoints) < 10:
                return JsonResponse({'success': False, 'message': 'Se necesitan al menos 10 capturas'})
            
            try:
                cap = cv2.VideoCapture(ip_port)
                if not cap.isOpened():
                    return JsonResponse({'success': False, 'message': 'No se pudo abrir el stream de la cámara para calibrar.'})

                ret, frame = cap.read()
                cap.release()

                if not ret or frame is None:
                    return JsonResponse({'success': False, 'message': 'No se pudo capturar la imagen para calibrar'})

                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                
                # Convertir las listas almacenadas de vuelta a numpy arrays
                objpoints = [np.array(pts, dtype=np.float32) for pts in objpoints]
                imgpoints = [np.array(pts, dtype=np.float32) for pts in imgpoints]
                
                ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
                
                if ret:
                    camera_data = {
                        "ip_url": ip_port,
                        "label": request.session.get('camera_name'),
                        "cam_matrix": mtx.tolist(),
                        "dist_coeff": dist.tolist(),
                    }
                    ambiente = request.session.get('ambiente')
                    piso = request.session.get('piso')
                    
                    if not ambiente or not piso:
                        return JsonResponse({'success': False, 'message': 'Falta información del ambiente o piso'})
                    
                    update_camera_data(ambiente, piso, camera_data)
                    
                    return JsonResponse({
                        'success': True,
                        'camera_matrix': mtx.tolist(),
                        'dist_coeffs': dist.tolist()
                    })
                else:
                    return JsonResponse({'success': False, 'message': 'La calibración falló'})
            except Exception as e:
                logger.exception(f"Error al calibrar: {str(e)}")
                return JsonResponse({'success': False, 'message': f'Error al calibrar: {str(e)}'})

    return render(request, 'calibrar_camara.html', context)


def calibrar_camara_paso3(request):
    context = {'step': '3'}
    ip_port = request.session.get('ip_port')
    context['ip_port'] = ip_port

    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_50)
    aruco_params = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(aruco_dict, aruco_params)

    marker_size = 0.15  # 15 cm

    if request.method == 'POST' and request.headers.get('X-Requested-With') == 'XMLHttpRequest':
        action = request.POST.get('action')
        logger.info(f"Paso 3 - Acción: {action}, URL del stream: {ip_port}")

        if action == 'capture_near' or action == 'capture_far':
            try:
                cap = cv2.VideoCapture(ip_port)
                if not cap.isOpened():
                    return JsonResponse({'success': False, 'message': 'No se pudo abrir el stream de la cámara.'})

                ret, frame = cap.read()
                cap.release()

                if not ret or frame is None:
                    return JsonResponse({'success': False, 'message': 'No se pudo capturar la imagen'})

                corners, ids, _ = detector.detectMarkers(frame)
                if ids is not None:
                    marker_width_px = np.linalg.norm(corners[0][0][1] - corners[0][0][0])
                    pixels_per_meter = marker_width_px / marker_size

                    if action == 'capture_near':
                        request.session['pixels_per_meter_near'] = request.session.get('pixels_per_meter_near', []) + [pixels_per_meter]
                    else:
                        request.session['pixels_per_meter_far'] = request.session.get('pixels_per_meter_far', []) + [pixels_per_meter]

                    return JsonResponse({'success': True, 'captures': len(request.session.get('pixels_per_meter_near', [])) if action == 'capture_near' else len(request.session.get('pixels_per_meter_far', []))})
                else:
                    return JsonResponse({'success': False, 'message': 'No se detectó el marcador ArUco'})

            except Exception as e:
                logger.exception(f"Error al capturar: {str(e)}")
                return JsonResponse({'success': False, 'message': f'Error al capturar: {str(e)}'})

        elif action == 'calculate':
            pixels_per_meter_near = request.session.get('pixels_per_meter_near', [])
            pixels_per_meter_far = request.session.get('pixels_per_meter_far', [])

            if len(pixels_per_meter_near) < 5 or len(pixels_per_meter_far) < 5:
                return JsonResponse({'success': False, 'message': 'Se necesitan al menos 5 capturas para cada distancia'})

            pixels_per_meter = np.mean(pixels_per_meter_near[:5] + pixels_per_meter_far[:5])
            
            request.session['pixels_per_meter'] = pixels_per_meter
            camera_data = load_json_data()
            for amb in camera_data["ambientes"]:
                if amb["nombreAmbiente"] == request.session.get('ambiente'):
                    for cam in amb["piso"][request.session.get('piso')]["camara"]:
                        if cam["ip_url"] == ip_port:
                            cam["px_m"] = pixels_per_meter
                            break
                    break
            save_json_data(camera_data)
            
            request.session['pixels_per_meter'] = pixels_per_meter
            
            return JsonResponse({'success': True, 'pixels_per_meter': pixels_per_meter})
        

    return render(request, 'calibrar_camara.html', context)


def capture_frame(ip_port):
    try:
        stream = urllib.request.urlopen(ip_port, timeout=5)
        bytes_data = b''
        for i in range(30):
            bytes_data += stream.read(1024)
            a = bytes_data.find(b'\xff\xd8')
            b = bytes_data.find(b'\xff\xd9')
            if a != -1 and b != -1:
                jpg = bytes_data[a:b+2]
                bytes_data = bytes_data[b+2:]
                frame = cv2.imdecode(np.frombuffer(jpg, dtype=np.uint8), cv2.IMREAD_COLOR)
                if frame is not None:
                    return frame
        return None
    except Exception as e:
        print(f"Error capturing frame: {str(e)}")
        return None


def gen_frames(camera_url):
    cap = cv2.VideoCapture(camera_url)
    if not cap.isOpened():
        print(f"No se pudo abrir el stream de la cámara: {camera_url}")
        return  # Opcional: podrías manejar esto de otra manera

    while True:
        success, frame = cap.read()
        if not success:
            print("No se pudo leer el frame del stream.")
            break

        # Opcional: Aquí puedes agregar cualquier procesamiento de imagen que necesites

        ret, buffer = cv2.imencode('.jpg', frame)
        if not ret:
            print("No se pudo codificar el frame a JPEG.")
            continue

        frame_bytes = buffer.tobytes()
        try:
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        except GeneratorExit:
            # Manejar el cierre de la conexión
            print("Conexión cerrada por el cliente.")
            break
        except Exception as e:
            print(f"Error al generar el frame: {e}")
            break

    cap.release()

def camera_stream(request):
    ip_port = request.GET.get('ip_port')
    if not ip_port:
        return HttpResponseBadRequest("El parámetro IP:Port es obligatorio")

    # Opcional: Validar el formato de la URL
    # Puedes agregar validaciones para asegurarte de que la URL es válida

    return StreamingHttpResponse(gen_frames(ip_port),
                                 content_type='multipart/x-mixed-replace; boundary=frame')

def calibrar_camara_paso_final(request):
    context = {
        'step': 'final',
        'camera_name': request.session.get('camera_name', ''),
        'ambiente': request.session.get('ambiente', ''),
        'piso': request.session.get('piso', ''),
        'pixels_per_meter': request.session.get('pixels_per_meter', ''),
    }
    return render(request, 'calibrar_camara.html', context)

def load_json_data():
    if os.path.exists(JSON_FILE_PATH):
        with open(JSON_FILE_PATH, 'r') as file:
            return json.load(file)
    return {"ambientes": []}

def save_json_data(data):
    os.makedirs(os.path.dirname(JSON_FILE_PATH), exist_ok=True)
    with open(JSON_FILE_PATH, 'w') as file:
        json.dump(data, file, indent=2)


def update_camera_data(ambiente, piso, camera_data):
    data = load_json_data()
    for amb in data["ambientes"]:
        if amb["nombreAmbiente"] == ambiente:
            if piso not in amb["piso"]:
                amb["piso"][piso] = {}
            if "camara" not in amb["piso"][piso]:
                amb["piso"][piso]["camara"] = []
            
            for cam in amb["piso"][piso]["camara"]:
                if cam["ip_url"] == camera_data["ip_url"]:
                    cam.update(camera_data)
                    break
            else:
                amb["piso"][piso]["camara"].append(camera_data)
            break
    else:
        new_ambiente = {
            "nombreAmbiente": ambiente,
            "piso": {
                piso: {
                    "camara": [camera_data]
                }
            }
        }
        data["ambientes"].append(new_ambiente)
    save_json_data(data)


#Seccion supervisar camara:

def supervisar_camara(request, camera_uuid):
    logger.info(f"Buscando cámara con UUID: {camera_uuid}")
    
    camera_data = get_camera_by_uuid(camera_uuid)
    if not camera_data:
        logger.error(f"Cámara con UUID {camera_uuid} no encontrada")
        return HttpResponse("Cámara no encontrada", status=404)

    context = {
        'camera': camera_data,
        'ambiente': camera_data.get('ambiente'),
        'piso': camera_data.get('piso'),
        'camera_uuid': camera_uuid,
    }
    logger.info(f"Cámara encontrada: {camera_data}")
    return render(request, 'supervisar_camara.html', context)


def stream_supervisar_camara(request, camera_uuid):
    camera_data = get_camera_by_uuid(camera_uuid)
    if not camera_data:
        return HttpResponse("Cámara no encontrada", status=404)

    ip_url = camera_data['ip_url']
    polygons = camera_data.get('mesas', [])

    def generate_frames():
        cap = cv2.VideoCapture(ip_url)
        while True:
            success, frame = cap.read()
            if not success:
                # Retry connection
                cap.release()
                time.sleep(5)
                cap = cv2.VideoCapture(ip_url)
                continue

            frame = cv2.resize(frame, (640, 480))

            # Draw polygons and labels
            for polygon in polygons:
                points = np.array(polygon['points'], dtype=np.int32)
                cv2.polylines(frame, [points], isClosed=True, color=polygon['color'], thickness=2)
                text_position = tuple(map(int, points[0]))
                cv2.putText(frame, polygon['label'], text_position,
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, polygon['color'], 2)

                # Display additional information
                info_text = f"Max: {polygon['max_personas']} | Detectadas: {polygon.get('personas_Detectadas', 0)}"
                cv2.putText(frame, info_text, (text_position[0], text_position[1] + 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, polygon['color'], 1)

            _, buffer = cv2.imencode('.jpg', frame)
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

        cap.release()

    return StreamingHttpResponse(generate_frames(), content_type='multipart/x-mixed-replace; boundary=frame')



def obtener_info_mesas_uuid(request, camera_uuid):
    db_data = load_json_data()
    camera = None
    for ambiente in db_data['ambientes']:
        for piso, piso_data in ambiente['piso'].items():
            for cam in piso_data.get('camara', []):
                if cam.get('uuid') == str(camera_uuid):
                    camera = cam
                    break
            if camera:
                break
        if camera:
            break
    if not camera:
        return JsonResponse({'error': 'Cámara no encontrada'})

    polygons = camera.get('mesas', [])

    return JsonResponse({'mesas': polygons})






