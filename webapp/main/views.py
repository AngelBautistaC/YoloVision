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
from .firebase_helpers import get_camera_by_ip, save_camera_data,get_camera_by_uuid,save_camera_data2,get_ambientes_and_pisos_from_db
from .tasks import start_detection_task
from .utils import expand_polygon, initialize_tracking, initialize_counting_regions, track_and_count

from django.shortcuts import render, redirect
from django.contrib import messages
from .forms import CreateUserForm
from .firebase_helpers import create_user_in_firebase  # Function to handle Firebase user creation

from django.shortcuts import render, redirect
from django.contrib import messages
from .forms import LoginForm
from .firebase_helpers import authenticate_firebase_user, save_user_to_firebase

logger = logging.getLogger(__name__)

from firebase_admin import credentials, db

from django.contrib.auth.decorators import login_required
from django.shortcuts import redirect,render

import json
import os
from django.conf import settings
from django.contrib.auth import login
from django.contrib.auth.models import User



from functools import wraps
from django.shortcuts import redirect
from django.conf import settings

def role_required(allowed_roles=[]):
    """
    Decorador para vistas que verifica si el usuario tiene uno de los roles permitidos.
    Si no es así, redirige a la página de acceso denegado.
    """
    def decorator(view_func):
        @wraps(view_func)
        def _wrapped_view(request, *args, **kwargs):
            user_role = request.session.get('user_role')
            if user_role in allowed_roles:
                return view_func(request, *args, **kwargs)
            else:
                return redirect('access_denied')
        return _wrapped_view
    return decorator


@login_required
@role_required(allowed_roles=['administrator'])
def admin_dashboard(request):

    return render(request, 'admin_dashboard.html')
   

@login_required
@role_required(allowed_roles=['operator'])
def operator_dashboard(request):
    
    return render(request, 'operator_dashboard.html')
    

@login_required
def user_dashboard(request):
    return render(request, 'user_dashboard.html')

# views.py

def access_denied(request):
    return render(request, 'access_denied.html')


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

JSON_FILE_PATH = os.path.join(settings.BASE_DIR, 'data', 'database.json')

def login_view(request):
    if request.method == 'POST':
        form = LoginForm(request.POST)
        if form.is_valid():
            email = form.cleaned_data['email']
            password = form.cleaned_data['password']
            user, role = authenticate_firebase_user(email, password)

            if user:
                # Set session data and create Django User instance if it doesn't exist
                django_user, created = User.objects.get_or_create(username=user.uid)
                django_user.is_staff = role == 'administrator'  # Corrected role check
                django_user.save()

                # Log in the user to establish session
                login(request, django_user)

                # Store the user's role in the session
                request.session['user_role'] = role

                # Redirect based on role
                if role == 'operator':
                    return redirect('operator_dashboard')
                elif role == 'user':
                    return redirect('user_dashboard')
                elif role == 'administrator':
                    return redirect('admin_dashboard')
            else:
                messages.error(request, 'Invalid credentials. Please try again.')
    else:
        form = LoginForm()

    return render(request, 'login.html', {'form': form})



@login_required
@role_required(allowed_roles=['operator', 'administrator'])
def configuracion_view(request):
    # Limpiar datos de calibración previos
    calibration_keys = ['ip_port', 'camera_name', 'ambiente', 'piso', 'objpoints', 'imgpoints',
                        'pixels_per_meter_near', 'pixels_per_meter_far', 'pixels_per_meter',
                        'camera_data', 'detections', 'manual_polygons', 'tracking_active']
    for key in calibration_keys:
        if key in request.session:
            del request.session[key]
    return render(request, 'configuracion.html')




@login_required
def crear_view(request):
    # Verifica si el rol del usuario está almacenado en la sesión
    user_role = request.session.get('user_role')

    # Solo permite el acceso si el rol es 'operator' o 'administrator'
    if user_role not in ['operator', 'administrator']:
        return redirect('access_denied')  # Redirige a una vista de acceso denegado

    if request.method == 'POST':
        form = CreateUserForm(request.POST)
        if form.is_valid():
            username = form.cleaned_data['username']
            email = form.cleaned_data['email']
            password = form.cleaned_data['password']
            role = form.cleaned_data['role']

            # Create user in Firebase and save additional data
            user_id = create_user_in_firebase(username, email, password, role)
            if user_id:
                messages.success(request, f'User {username} created successfully!')
                return redirect('monitoring_view')
            else:
                messages.error(request, 'There was an error creating the user.')
    else:
        form = CreateUserForm()

    return render(request, 'crear.html', {'form': form})
    
# views.py
from django.shortcuts import render, redirect
from django.contrib import messages

def crear_form_view(request):
    # Verifica si el rol del usuario está almacenado en la sesión
    user_role = request.session.get('user_role')

    # Solo permite el acceso si el rol es 'operator' o 'administrator'
    if user_role not in ['operator', 'administrator']:
        return redirect('access_denied')  # Redirige a una vista de acceso denegado

    if request.method == 'POST':
        username = request.POST.get('username')
        email = request.POST.get('email')
        password = request.POST.get('password')
        role = request.POST.get('role')
        
        print("Attempting to create user in Firebase...")
        
        user_id = create_user_in_firebase(username, email, password, role)
        
        if user_id:
            print("User created successfully.")
            # Display a success message or render a success template
            messages.success(request, "User created successfully.")
        else:
            print("User creation failed.")
            # Display an error message or handle failure
            messages.error(request, "Failed to create user.")
        
        return redirect('crear_form')  # Modify with the actual name of your URL if different

    return render(request, 'crear.html')




logger = logging.getLogger(__name__)

# def calibrar_camara_paso1(request):
#         # Limpiar datos de calibración previos
#     calibration_keys = ['ip_port', 'camera_name', 'ambiente', 'piso', 'objpoints', 'imgpoints', 'pixels_per_meter_near', 'pixels_per_meter_far', 'pixels_per_meter']
#     for key in calibration_keys:
#         if key in request.session:
#             del request.session[key]
    
    
    
#     context = {'step': '1', 'camera_connected': False}

#     if request.method == 'POST' and request.headers.get('X-Requested-With') == 'XMLHttpRequest':
        
#         ip_port = request.POST.get('ip_port')
#         camera_name = request.POST.get('camera_name')
#         ambiente = request.POST.get('ambiente')
#         piso = request.POST.get('piso', '1')  # Asumimos piso 1 por defecto

#         if not ip_port:
#             return JsonResponse({'success': False, 'message': 'La URL del stream está vacía'})

#         try:
#             cap = cv2.VideoCapture(ip_port)
#             if not cap.isOpened():
#                 logger.error(f"No se pudo abrir el stream: {ip_port}")
#                 return JsonResponse({'success': False, 'message': 'No se pudo abrir el stream de la cámara. Verifica la URL y el formato.'})

#             frame_read = False
#             for _ in range(30):
#                 ret, frame = cap.read()
#                 if ret and frame is not None:
#                     frame_read = True
#                     break

#             cap.release()
#             frame = cv2.resize(frame, (640, 480))

#             if frame_read:
#                 # Save necessary data in session
#                 request.session['ip_port'] = ip_port
#                 request.session['camera_name'] = camera_name
#                 request.session['ambiente'] = ambiente
#                 request.session['piso'] = piso

#                 # Create camera_data dictionary
#                 camera_data = {
#                     "ip_url": ip_port,
#                     "label": camera_name,
#                 }

#                 # Save camera data to JSON
#                 update_camera_data(ambiente, piso, camera_data)

#                 logger.info(f"Conexión exitosa. Datos guardados en sesión y JSON.")
#                 return JsonResponse({'success': True, 'message': 'Conexión exitosa'})
#             else:
#                 logger.error("No se pudo leer un frame después de 30 intentos")
#                 return JsonResponse({'success': False, 'message': 'No se pudo leer un frame de la cámara después de 30 intentos'})
#         except Exception as e:
#             logger.exception(f"Error inesperado al conectar con la cámara: {str(e)}")
#             return JsonResponse({'success': False, 'message': f'Error inesperado: {str(e)}'})

#     return render(request, 'calibrar_camara.html', context)


# @login_required
# @role_required(allowed_roles=['operator', 'administrator'])
# def calibrar_camara_paso3(request):

#     context = {'step': '3'}
#     ip_port = request.session.get('ip_port')
#     context['ip_port'] = ip_port

#     # Verificar si se ha completado el paso 1
#     if not request.session.get('ip_port'):
#         return redirect('calibrar_camara_paso1')
    
#     aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_50)
#     aruco_params = cv2.aruco.DetectorParameters()
#     detector = cv2.aruco.ArucoDetector(aruco_dict, aruco_params)

#     marker_size = 0.15  # 15 cm

#     if request.method == 'POST' and request.headers.get('X-Requested-With') == 'XMLHttpRequest':
#         action = request.POST.get('action')
#         logger.info(f"Paso 3 - Acción: {action}, URL del stream: {ip_port}")

#         if action == 'capture_near' or action == 'capture_far':
#             try:
#                 cap = cv2.VideoCapture(ip_port)
#                 if not cap.isOpened():
#                     return JsonResponse({'success': False, 'message': 'No se pudo abrir el stream de la cámara.'})

#                 ret, frame = cap.read()
#                 cap.release()

#                 if not ret or frame is None:
#                     return JsonResponse({'success': False, 'message': 'No se pudo capturar la imagen'})
                
#                 # Redimensionar el frame a 640x480
#                 frame = cv2.resize(frame, (640, 480))
#                 corners, ids, _ = detector.detectMarkers(frame)
#                 if ids is not None:
#                     marker_width_px = np.linalg.norm(corners[0][0][1] - corners[0][0][0])
#                     pixels_per_meter = marker_width_px / marker_size

#                     if action == 'capture_near':
#                         request.session['pixels_per_meter_near'] = request.session.get('pixels_per_meter_near', []) + [pixels_per_meter]
#                     else:
#                         request.session['pixels_per_meter_far'] = request.session.get('pixels_per_meter_far', []) + [pixels_per_meter]

#                     return JsonResponse({'success': True, 'captures': len(request.session.get('pixels_per_meter_near', [])) if action == 'capture_near' else len(request.session.get('pixels_per_meter_far', []))})
#                 else:
#                     return JsonResponse({'success': False, 'message': 'No se detectó el marcador ArUco'})

#             except Exception as e:
#                 logger.exception(f"Error al capturar: {str(e)}")
#                 return JsonResponse({'success': False, 'message': f'Error al capturar: {str(e)}'})

#         elif action == 'calculate':
#             pixels_per_meter_near = request.session.get('pixels_per_meter_near', [])
#             pixels_per_meter_far = request.session.get('pixels_per_meter_far', [])

#             if len(pixels_per_meter_near) < 5 or len(pixels_per_meter_far) < 5:
#                 return JsonResponse({'success': False, 'message': 'Se necesitan al menos 5 capturas para cada distancia'})

#             pixels_per_meter = np.mean(pixels_per_meter_near[:5] + pixels_per_meter_far[:5])
            
#             request.session['pixels_per_meter'] = pixels_per_meter
#             camera_data = load_json_data()
#             for amb in camera_data["ambientes"]:
#                 if amb["nombreAmbiente"] == request.session.get('ambiente'):
#                     for cam in amb["piso"][request.session.get('piso')]["camara"]:
#                         if cam["ip_url"] == ip_port:
#                             cam["px_m"] = pixels_per_meter
#                             break
#                     break
#             save_json_data(camera_data)
            
#             request.session['pixels_per_meter'] = pixels_per_meter
            
#             return JsonResponse({'success': True, 'pixels_per_meter': pixels_per_meter})
        

#     return render(request, 'calibrar_camara.html', context)

@login_required
@role_required(allowed_roles=['operator', 'administrator'])
def calibrar_camara_paso1(request):
    # Limpiar datos de calibración previos
    calibration_keys = ['ip_port', 'camera_name', 'ambiente', 'piso', 'pixels_per_meter_near', 'pixels_per_meter_far', 'pixels_per_meter']
    for key in calibration_keys:
        if key in request.session:
            del request.session[key]

    # Obtener ambientes y pisos desde la base de datos
    ambientes_data = get_ambientes_and_pisos_from_db()
    ambientes_data_json = json.dumps(ambientes_data)

    context = {
        'step': '1',
        'camera_connected': False,
        'ambientes_data': ambientes_data,
        'ambientes_data_json': ambientes_data_json,
    }

    
    if request.method == 'POST' and request.headers.get('X-Requested-With') == 'XMLHttpRequest':
        ip_port = request.POST.get('ip_port')
        camera_name = request.POST.get('camera_name')
        ambiente = request.POST.get('ambiente')
        piso = request.POST.get('piso')

        if not ip_port:
            return JsonResponse({'success': False, 'message': 'La URL del stream está vacía'})

        # Verificar si la cámara ya existe
        camera_uuid, existing_camera_data = get_camera_by_ip(ip_port)
        if camera_uuid:
            return JsonResponse({'success': False, 'message': 'Ya existe una cámara con esta IP:Puerto en la base de datos. No puede calibrar una cámara con la misma IP:Puerto.'})

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
                # Guardar datos en la sesión
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

@login_required
@role_required(allowed_roles=['operator', 'administrator'])
def calibrar_camara_paso3(request):
    context = {'step': '3'}
    ip_port = request.session.get('ip_port')
    camera_name = request.session.get('camera_name')
    ambiente = request.session.get('ambiente')
    piso = request.session.get('piso')

    context['ip_port'] = ip_port

    # Verificar si se ha completado el paso 1
    if not ip_port:
        return redirect('calibrar_camara_paso1')

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

                # Redimensionar el frame a 640x480
                frame = cv2.resize(frame, (640, 480))
                corners, ids, _ = detector.detectMarkers(frame)
                if ids is not None and len(ids) > 0:
                    marker_width_px = np.linalg.norm(corners[0][0][1] - corners[0][0][0])
                    pixels_per_meter = marker_width_px / marker_size

                    if action == 'capture_near':
                        request.session['pixels_per_meter_near'] = request.session.get('pixels_per_meter_near', []) + [pixels_per_meter]
                        captures = len(request.session['pixels_per_meter_near'])
                    else:
                        request.session['pixels_per_meter_far'] = request.session.get('pixels_per_meter_far', []) + [pixels_per_meter]
                        captures = len(request.session['pixels_per_meter_far'])

                    return JsonResponse({'success': True, 'captures': captures})
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

            # Verificar nuevamente si la cámara ya existe
            camera_uuid, existing_camera_data = get_camera_by_ip(ip_port)
            if camera_uuid:
                return JsonResponse({'success': False, 'message': 'Ya existe una cámara con esta IP:Puerto en la base de datos. No puede calibrar una cámara con la misma IP:Puerto.'})

            # Crear el diccionario de datos de la cámara
            camera_uuid = str(uuid.uuid4())
            camera_data = {
                "ip_url": ip_port,
                "label": camera_name,
                "px_m": pixels_per_meter,
                "status": "calibrated",
                "uuid": camera_uuid,
            }

            # Guardar en Firebase
            save_camera_data(ambiente, piso, camera_uuid, camera_data)

            # Limpiar la sesión si es necesario
            calibration_keys = ['ip_port', 'camera_name', 'ambiente', 'piso', 'pixels_per_meter_near', 'pixels_per_meter_far', 'pixels_per_meter']
            for key in calibration_keys:
                if key in request.session:
                    del request.session[key]

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

        # Redimensionar el frame a 640x480
        frame = cv2.resize(frame, (640, 480))

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


@login_required
@role_required(allowed_roles=['operator', 'administrator'])
def calibrar_camara_paso_final(request):
    # Verificar si se ha completado el paso 3
    if not request.session.get('pixels_per_meter'):
        return redirect('calibrar_camara_paso1')

    context = {
        'step': 'final',
        'camera_name': request.session.get('camera_name', ''),
        'ambiente': request.session.get('ambiente', ''),
        'piso': request.session.get('piso', ''),
        'pixels_per_meter': request.session.get('pixels_per_meter', ''),
    }
    
        # Limpiar datos de calibración
    calibration_keys = ['ip_port', 'camera_name', 'ambiente', 'piso', 'objpoints', 'imgpoints', 'pixels_per_meter_near', 'pixels_per_meter_far', 'pixels_per_meter']
    for key in calibration_keys:
        if key in request.session:
            del request.session[key]
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


def supervisar_camara(request, camera_uuid):
    logger.info(f"Buscando cámara con UUID: {camera_uuid}")
    
    camera_data = get_camera_by_uuid(camera_uuid)
    if not camera_data:
        logger.error(f"Cámara con UUID {camera_uuid} no encontrada")
        return HttpResponse("Cámara no encontrada", status=404)

    context = {
        'camera': camera_data,
        'ambiente': camera_data.get('ambiente', 'N/A'),
        'piso': camera_data.get('piso', 'N/A'),
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

                # Ensure color is a tuple of integers
                color_data = polygon['color']
                if isinstance(color_data, list):
                    color = tuple(map(int, color_data))
                elif isinstance(color_data, str):
                    # Assuming the color is stored as a string like "(255, 0, 0)"
                    color = tuple(map(int, color_data.strip('()').split(',')))
                else:
                    # Default color if format is unexpected
                    color = (0, 255, 0)

                cv2.polylines(frame, [points], isClosed=True, color=color, thickness=2)
                text_position = tuple(map(int, points[0]))
                cv2.putText(frame, polygon.get('label', 'Mesa'), text_position,
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                # Display additional information
                max_personas = polygon.get('max_personas', 0)
                personas_detectadas = polygon.get('personas_Detectadas', 0)
                info_text = f"Max: {max_personas} | Detectadas: {personas_detectadas}"
                cv2.putText(frame, info_text, (text_position[0], text_position[1] + 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

            _, buffer = cv2.imencode('.jpg', frame)
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

        cap.release()

    return StreamingHttpResponse(generate_frames(), content_type='multipart/x-mixed-replace; boundary=frame')


def obtener_info_mesas_uuid(request, camera_uuid):
    camera_data = get_camera_by_uuid(camera_uuid)
    if not camera_data:
        return JsonResponse({'error': 'Cámara no encontrada'})

    polygons = camera_data.get('mesas', [])
    return JsonResponse({'mesas': polygons})

from collections import defaultdict



def obtener_datos_monitoreo():
    ambientes_ref = db.reference('/ambientes')
    ambientes_data = ambientes_ref.get()
    datos_monitoreo = []
    resumen_ambientes = defaultdict(lambda: {'personas_detectadas': 0, 'max_personas': 0, 'asientos_disponibles': 0})

    if isinstance(ambientes_data, dict):
        for ambiente_nombre, ambiente_data in ambientes_data.items():
            pisos = ambiente_data.get('piso', {})
            if isinstance(pisos, dict):
                for piso, piso_data in pisos.items():
                    camaras = piso_data.get('camara', {})
                    if isinstance(camaras, dict):
                        for cam_uuid, cam_data in camaras.items():
                            cam_label = cam_data.get('label', 'Sin etiqueta')
                            mesas = cam_data.get('mesas', [])
                            if isinstance(mesas, list):
                                for mesa in mesas:
                                    personas_detectadas = mesa.get("personas_Detectadas", 0)
                                    max_personas = mesa.get("max_personas", 0)
                                    asientos_disponibles = max_personas - personas_detectadas

                                    mesa_info = {
                                        "ambiente": ambiente_nombre,
                                        "piso": piso,
                                        "camara": cam_label,
                                        "mesa_id": mesa.get("id", "N/A"),
                                        "max_personas": max_personas,
                                        "personas_detectadas": personas_detectadas,
                                        "asientos_disponibles": asientos_disponibles,
                                    }
                                    datos_monitoreo.append(mesa_info)

                                    # Agregar al resumen
                                    resumen_ambientes[ambiente_nombre]['personas_detectadas'] += personas_detectadas
                                    resumen_ambientes[ambiente_nombre]['max_personas'] += max_personas
                                    resumen_ambientes[ambiente_nombre]['asientos_disponibles'] += asientos_disponibles

    return datos_monitoreo, dict(resumen_ambientes)

@login_required
def monitoring_view(request):
    datos_monitoreo, resumen_ambientes = obtener_datos_monitoreo()  # Llamar a la función correcta
    return render(request, 'monitoreo.html', {'datos_monitoreo': datos_monitoreo, 'resumen_ambientes': resumen_ambientes})
