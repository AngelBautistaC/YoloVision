from collections import defaultdict
import re
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
from .firebase_helpers import delete_camera_from_firebase,get_camera_by_ip, save_camera_data,get_camera_by_uuid,save_camera_data2,get_ambientes_and_pisos_from_db
# from .tasks import start_detection_task
# from .utils import expand_polygon, initialize_tracking, initialize_counting_regions, track_and_count
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
from logger.utils import create_log_entry
from logger.views import log_view
from logger.models import LogEntry
from django.urls import reverse # Para el middleware
from django.contrib.auth import logout
from django.views.decorators.cache import never_cache
from collections import defaultdict
# views.py
from django.shortcuts import render, redirect
from django.contrib import messages
from firebase_admin import db





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



# views.py

def access_denied(request):
    return render(request, 'access_denied.html')

def configurar_deteccion(request):
    ref = db.reference('ambientes')
    snapshot = ref.get()

    camaras_calibradas = []
    if snapshot:
        for ambiente_nombre, ambiente_data in snapshot.items():
            pisos_dict = ambiente_data.get('piso', {})
            if not isinstance(pisos_dict, dict):
                continue

            for piso_nombre, piso_data in pisos_dict.items():
                if not isinstance(piso_data, dict):
                    continue

                camaras_data = piso_data.get('camara', {})
                if not isinstance(camaras_data, dict):
                    continue

                for cam_uuid, cam_data in camaras_data.items():
                    # Mostrar solo cámaras con estado "calibrated"
                    if cam_data.get('status') == 'calibrated':
                        camaras_calibradas.append({
                            'ambiente': ambiente_nombre,
                            'piso': piso_nombre,
                            'label': cam_data.get('label', 'Sin etiqueta'),
                            'ip_url': cam_data['ip_url']
                        })

    context = {'camaras': camaras_calibradas}
    return render(request, 'configurar_deteccion.html', context)




def iniciar_deteccion(request):
    if request.method == 'POST':
        ip_url = request.POST.get('ip_url')
        camera_uuid, cam_data = get_camera_by_ip(ip_url)

        if not camera_uuid or not cam_data:
            return JsonResponse({'error': 'Cámara no encontrada'})

        # Guardar datos de la cámara en la sesión
        request.session['camera_data'] = cam_data
        # Set tracking_active to False
        request.session['tracking_active'] = False

        return JsonResponse({'success': True})
    
    return JsonResponse({'error': 'Método no permitido'})


def obtener_detecciones(request):
    detected_polygons = request.session.get('detections', [])
    manual_polygons = request.session.get('manual_polygons', [])
    all_polygons = detected_polygons + manual_polygons
    return JsonResponse({'detections': all_polygons})


#Deteccion de mesas -- FUNCIONAL
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

        model = YOLO('yolo11s-seg.pt')

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
        detection_interval = data.get('detection_interval', 5)

        camera_data = request.session.get('camera_data')
        if not camera_data:
            return JsonResponse({'error': 'No hay datos de cámara en la sesión'})

        ip_url = camera_data['ip_url']

        camera_uuid, existing_camera_data = get_camera_by_ip(ip_url)
        if not camera_uuid or not existing_camera_data:
            return JsonResponse({'error': 'Cámara no encontrada en Firebase'})

        existing_camera_data['mesas'] = polygons
        existing_camera_data['detection_interval'] = detection_interval
        existing_camera_data['status'] = 'configurated'
        if 'connection_state' not in existing_camera_data:
            existing_camera_data['connection_state'] = 'stable'
        if 'connection_attempts' not in existing_camera_data:
            existing_camera_data['connection_attempts'] = 0
        if 'last_detection_time' not in existing_camera_data:
            existing_camera_data['last_detection_time'] = 0

        ambiente_nombre = existing_camera_data['ambiente']
        piso = existing_camera_data['piso']

        save_camera_data2(ambiente_nombre, piso, camera_uuid, existing_camera_data)

        create_log_entry(request, 'Configurar Cámara', f'Finalizó la configuración de la cámara con IP: {ip_url}')

        # Ya no llamamos start_detection_task.delay(camera_uuid)
        # Las tareas periódicas de celery beat se encargarán.
        
        return JsonResponse({'success': True, 'camera_uuid': camera_uuid})

    return JsonResponse({'error': 'Método no permitido'})




def stream_video(request):
    camera_data = request.session.get('camera_data')
    if not camera_data:
        return JsonResponse({'error': 'No hay datos de cámara en la sesión'})

    ip_url = camera_data['ip_url']
    pixels_per_meter = camera_data['px_m']
    tracking_active = request.session.get('tracking_active', False)

    # Obtener todos los polígonos desde la cámara y/o la sesión, tal como se hacía en local
    polygons = camera_data.get('mesas', [])
    detections = request.session.get('detections', [])
    manual_polygons = request.session.get('manual_polygons', [])
    polygons = polygons + detections + manual_polygons

    def generate_frames():
        cap = cv2.VideoCapture(ip_url)
        if not cap.isOpened():
            return

        if tracking_active:
            model = YOLO('yolo11n.pt')
        else:
            model = None

        while True:
            success, frame = cap.read()
            if not success:
                break
            frame = cv2.resize(frame, (640, 480))

            # Si no hay tracking, personas_detectadas = 0
            if not tracking_active or model is None:
                for polygon in polygons:
                    polygon['personas_Detectadas'] = 0

            # Dibujar polígonos directamente desde los datos en sesión
            for polygon in polygons:
                points = np.array(polygon['points'], dtype=np.int32)
                color = polygon.get('color', (0, 255, 0))
                cv2.polylines(frame, [points], isClosed=True, color=color, thickness=2)
                text_position = tuple(map(int, points[0]))
                cv2.putText(frame, polygon['label'], text_position,
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                area = polygon.get('area', 0)
                max_p = polygon.get('max_personas', 0)
                detectadas = polygon.get('personas_Detectadas', 0)
                info_text = f"Area: {area:.2f}m² | Max: {max_p} | Detectadas: {detectadas}"
                cv2.putText(frame, info_text, (text_position[0], text_position[1] + 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

            ret, buffer = cv2.imencode('.jpg', frame)
            if not ret:
                continue
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

    # Obtener polígonos detectados en la sesión
    detected_polygons = request.session.get('detections', [])
    # Obtener polígonos manuales en la sesión
    manual_polygons = request.session.get('manual_polygons', [])

    # En el escenario local que funcionaba, no se dependía de backend_polygons cargados desde archivo.
    # Simplemente se dejaba vacío o se usaba lo que estuviera en la sesión.
    backend_polygons = []

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



def logout_view(request):
    logout(request)
    return redirect('login')



def login_view(request):

    if request.user.is_authenticated:
        return redirect('monitoreo')  # Redirige a la página de monitoreo si ya está logueado

    if request.method == 'POST':
        form = LoginForm(request.POST)
        if form.is_valid():
            email = form.cleaned_data['email']
            password = form.cleaned_data['password']
            user, role = authenticate_firebase_user(email, password)

            if user:
       
                firebase_token = user.firebase_token # Assuming authenticate_firebase_user now returns user with firebase_token

                # Store user's email in session
                request.session['user_email'] = user.email
                request.session.save()  # Guardar la sesión explícitamente

                # Create Django User instance if it doesn't exist
                django_user, created = User.objects.get_or_create(username=user.email)
                django_user.is_staff = role == 'admin'
                django_user.save()
                login(request, django_user)
                request.session['user_role'] = role
                request.session.modified = True  # Forzar actualización de la sesión
                request.session.save()  # Guardar la sesión explícitamente
                print(f"login_view: Rol guardado en sesión: {role}") # <-- Depuración

                create_log_entry(request, 'Inicio de sesión', f'Usuario: {user.email}')
                next_url = request.session.pop('next_url', None)
                #return redirect(next_url or 'monitoreo')
                response = redirect(next_url or 'monitoreo')
                response.set_cookie('firebase_token', firebase_token.decode('utf-8')) # Ensure token is string and set cookie
                return response

            else:
                logout(request)
                messages.error(request, 'Invalid credentials. Please try again.')
    else:
        form = LoginForm()

    return render(request, 'login.html', {'form': form})

@login_required
#@role_required(allowed_roles=['operator', 'administrator'])
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
    if user_role not in ['operator', 'admin']:
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
    

def crear_form_view(request):
    # Verifica si el rol del usuario está almacenado en la sesión
    user_role = request.session.get('user_role')

    # Solo permite el acceso si el rol es 'operator' o 'administrator'
    if user_role not in ['operator', 'admin']:
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


#------------------------------------------------------------------------------------------------
@login_required
def calibrar_camara_paso1(request):
    # Limpiar datos de calibración previos
    calibration_keys = ['ip_port', 'camera_name', 'ambiente', 'piso', 'pixels_per_meter_near',
                        'pixels_per_meter_far', 'pixels_per_meter', 'username', 'password', 'protocol']
    for key in calibration_keys:
        if key in request.session:
            del request.session[key]

    # Obtener ambientes y pisos
    ambientes_data = get_ambientes_and_pisos_from_db()
    ambientes_data_json = json.dumps(ambientes_data)

    context = {
        'step': '1',
        'camera_connected': False,
        'ambientes_data': ambientes_data,
        'ambientes_data_json': ambientes_data_json,
        'protocol_options': ['http', 'rtsp'],  # Opciones para el protocolo
    }

    if request.method == 'POST' and request.headers.get('X-Requested-With') == 'XMLHttpRequest':
        ip_port = request.POST.get('ip_port')
        camera_name = request.POST.get('camera_name')
        ambiente = request.POST.get('ambiente')
        piso = request.POST.get('piso')
        username = request.POST.get('username')  # Get username
        password = request.POST.get('password')  # Get password
        protocol = request.POST.get('protocol', 'http') # Get protocol, default to http

        if not ip_port:
            return JsonResponse({'success': False, 'message': 'La URL del stream está vacía'})

        # --- Input Validation (CRUCIAL) ---
        # Basic URL structure validation (this is NOT exhaustive, but catches common errors)
        if not re.match(r"^(https?|rtsp)://", ip_port, re.IGNORECASE):
            # Allow URLs *without* protocol if username/password are given.
            if not username and not password:
                return JsonResponse({'success': False, 'message': 'La URL debe comenzar con http://, https://, o rtsp://'})

        # Check for duplicate cameras.
        camera_uuid, existing_camera_data = get_camera_by_ip(ip_port)
        if camera_uuid:
            return JsonResponse({'success': False,
                                 'message': 'Ya existe una cámara con esta IP/URL en la base de datos.'})

        # Construct the full URL *correctly*.
        if username and password:
            # Remove any existing protocol://
            ip_port = re.sub(r"^(https?|rtsp)://", "", ip_port, flags=re.IGNORECASE)
            full_url = f"{protocol.lower()}://{username}:{password}@{ip_port}"
        else:
            # If no username/password, use the provided ip_port directly.
            full_url = ip_port
            if not (full_url.lower().startswith("http://") or full_url.lower().startswith("https://") or full_url.lower().startswith("rtsp://")):
                full_url = f"{protocol.lower()}://{full_url}" # Add protocol if missing.
        
        print(f"Full URL constructed: {full_url}") # Debug print

        try:
            cap = cv2.VideoCapture(full_url)
            if not cap.isOpened():
                logger.error(f"No se pudo abrir el stream: {full_url}")
                return JsonResponse({'success': False,
                                     'message': 'No se pudo abrir el stream.  Verifica la URL, las credenciales y que la cámara sea accesible.'})

            frame_read = False
            for _ in range(30):  # Try to read frames
                ret, frame = cap.read()
                if ret and frame is not None:
                    frame_read = True
                    break

            cap.release()

            if frame_read:
                # Store data in session.  Store the *provided* ip_port, NOT the full_url.
                request.session['ip_port'] = ip_port
                request.session['camera_name'] = camera_name
                request.session['ambiente'] = ambiente
                request.session['piso'] = piso
                request.session['username'] = username  # Store username
                request.session['password'] = password  # Store password
                request.session['protocol'] = protocol # Store Protocol.

                logger.info(f"Conexión exitosa. URL: {full_url}, IP_PORT: {ip_port}")
                return JsonResponse({'success': True, 'message': 'Conexión exitosa'})
            else:
                logger.error("No se pudo leer un frame después de 30 intentos")
                return JsonResponse({'success': False,
                                     'message': 'No se pudo leer un frame de la cámara después de varios intentos'})

        except Exception as e:
            logger.exception(f"Error inesperado: {str(e)}")
            return JsonResponse({'success': False, 'message': f'Error inesperado: {str(e)}'})

    return render(request, 'calibrar_camara.html', context)


def camera_stream(request):
    ip_port = request.GET.get('ip_port')  # Get the *original* ip_port
    username = request.session.get('username')
    password = request.session.get('password')
    protocol = request.session.get('protocol', 'http')

    if not ip_port:
        return HttpResponseBadRequest("El parámetro 'ip_port' es obligatorio")

    # Reconstruct full_url (SAME LOGIC AS ABOVE!)
    if username and password:
        # Remove any existing protocol://
        ip_port = re.sub(r"^(https?|rtsp)://", "", ip_port, flags=re.IGNORECASE)
        full_url = f"{protocol.lower()}://{username}:{password}@{ip_port}"
    else:
        full_url = ip_port
        if not (full_url.lower().startswith("http://") or full_url.lower().startswith("https://") or full_url.lower().startswith("rtsp://")):
            full_url = f"{protocol.lower()}://{full_url}"

    print(f"Usando URL para streaming: {full_url}")
    return StreamingHttpResponse(gen_frames(full_url),
                                 content_type='multipart/x-mixed-replace; boundary=frame')


@login_required
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
    calibration_keys = ['ip_port', 'camera_name', 'ambiente', 'piso', 'objpoints', 'imgpoints', 'pixels_per_meter_near', 'pixels_per_meter_far', 'pixels_per_meter','username','password','protocol']
    for key in calibration_keys:
        if key in request.session:
            del request.session[key]
    return render(request, 'calibrar_camara.html', context)


def gen_frames(camera_url, max_retries=3, retry_delay=2):
    """
    Genera frames de video desde una cámara IP, manejando reconexiones.

    Args:
        camera_url (str): URL completa del stream de la cámara (incluyendo protocolo, usuario/contraseña si es necesario).
        max_retries (int): Número máximo de intentos de reconexión.
        retry_delay (int):  Tiempo de espera en segundos entre intentos de reconexión.

    Yields:
        bytes:  Un frame de video codificado en JPEG, en formato multipart, o None si la conexión falla permanentemente.
    """

    retries = 0
    cap = None  # Inicializa cap fuera del bucle

    while retries <= max_retries:
        try:
            if cap is None or not cap.isOpened(): #Verifica si cap está definida e inicializada.
                cap = cv2.VideoCapture(camera_url)
                if not cap.isOpened():
                    raise cv2.error("Could not open video stream")  #Lanza una excepción de opencv

            while True:
                success, frame = cap.read()
                if not success or frame is None:
                    raise cv2.error("Could not read frame")

                frame = cv2.resize(frame, (640, 480))
                ret, buffer = cv2.imencode('.jpg', frame)
                if not ret:
                    raise cv2.error("Could not encode frame to JPEG")

                frame_bytes = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
                # Reset retries on successful frame read
                retries = 0

        except cv2.error as e:  # Captura excepciones específicas de OpenCV
            print(f"Error de OpenCV: {e}")
            retries += 1
            if retries > max_retries:
                print("Número máximo de intentos de conexión alcanzado.  Abortando.")
                break  # Sale del bucle principal si se supera max_retries.
            print(f"Intentando reconectar en {retry_delay} segundos... (Intento {retries}/{max_retries})")
            time.sleep(retry_delay)

        except GeneratorExit:
            print("Conexión cerrada por el cliente.")
            break  # Importante: salir si el cliente cierra la conexión

        except Exception as e:
            print(f"Error inesperado: {e}")
            break
        finally:
          if cap is not None:
            cap.release()



# views.py  (ONLY the modified and new functions)

@login_required
def calibrar_camara_paso3(request):
    context = {'step': '3'}
    ip_port = request.session.get('ip_port')  # Get the *original* ip_port
    camera_name = request.session.get('camera_name')
    ambiente = request.session.get('ambiente')
    piso = request.session.get('piso')
    username = request.session.get('username')
    password = request.session.get('password')
    protocol = request.session.get('protocol', 'http')

    context['ip_port'] = ip_port  # Pass the original ip_port to the template

    if not ip_port:
        return redirect('calibrar_camara_paso1')

    # Reconstruct full_url, using the *same logic* as paso1.  This is important!
    if username and password:
        # Remove any existing protocol://
        ip_port = re.sub(r"^(https?|rtsp)://", "", ip_port, flags=re.IGNORECASE)
        full_url = f"{protocol.lower()}://{username}:{password}@{ip_port}"
    else:
        full_url = ip_port
        if not (full_url.lower().startswith("http://") or full_url.lower().startswith("https://") or full_url.lower().startswith("rtsp://")):
            full_url = f"{protocol.lower()}://{full_url}" # Add protocol if missing

    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_50)
    aruco_params = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(aruco_dict, aruco_params)
    marker_size = 0.15  # 15 cm

    if request.method == 'POST' and request.headers.get('X-Requested-With') == 'XMLHttpRequest':
        action = request.POST.get('action')
        logger.info(f"Paso 3 - Acción: {action}, URL: {full_url}")  # Log the *full* URL

        if action == 'capture_near' or action == 'capture_far':
            # ... (Rest of your Paso 3 capture logic, using full_url) ...  (No changes here)
            try:
                cap = cv2.VideoCapture(full_url)
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

        elif action == 'calculate':  # Handles *both* ArUco and line calculation
            # --- ArUco Calculation ---
            pixels_per_meter_near = request.session.get('pixels_per_meter_near', [])
            pixels_per_meter_far = request.session.get('pixels_per_meter_far', [])

            if len(pixels_per_meter_near) >= 1 and len(pixels_per_meter_far) >= 1:
                pixels_per_meter = np.mean(pixels_per_meter_near[:1] + pixels_per_meter_far[:1])
                request.session['pixels_per_meter'] = pixels_per_meter  # Store for final step

                # --- Store Camera Data (using ArUco result) ---
                camera_uuid, existing_camera_data = get_camera_by_ip(ip_port)
                if camera_uuid:  # Check for duplicate
                    return JsonResponse({'success': False, 'message': 'Ya existe una cámara con esta IP/URL en la base de datos.'})

                camera_uuid = str(uuid.uuid4())
                camera_data = {
                    "ip_url": ip_port,  # Store the *original* ip_port
                    "label": camera_name,
                    "px_m": pixels_per_meter,  # Use ArUco-calculated value
                    "status": "calibrated",
                    "uuid": camera_uuid,
                    "username": username,
                    "password": password,
                    "protocol": protocol,
                    "full_url": full_url  # Include full_url
                }

                save_camera_data(ambiente, piso, camera_uuid, camera_data)
                create_log_entry(request, 'Calibrar Cámara', f'Finalizó calibración (ArUco) en cámara: {ip_port}')
                return JsonResponse({'success': True, 'message': 'Calibración finalizada', 'pixels_per_meter': pixels_per_meter})
            else:
                 return JsonResponse({'success': False, 'message': 'Se necesitan al menos 1 capturas para cada distancia'}) #MODIFIED


        elif action == 'calculate_line':  # New action for line-based calculation
            line_length_str = request.POST.get('line_length')  # Get line length as string
            line_points = request.session.get('line_points')

            # --- Input Validation ---
            if not line_length_str:
                return JsonResponse({'success': False, 'message': 'Debe ingresar la longitud de la línea.'})
            try:
                line_length = float(line_length_str)  # Convert to float
                if line_length <= 0:
                    raise ValueError("Length must be positive")
            except ValueError:
                return JsonResponse({'success': False, 'message': 'La longitud de la línea debe ser un número positivo.'})

            if not line_points or len(line_points) != 2:
                return JsonResponse({'success': False, 'message': 'Debe dibujar una línea de dos puntos en la imagen.'})


            # Calculate pixels per meter
            p1 = np.array(line_points[0])  # First point as numpy array
            p2 = np.array(line_points[1])  # Second point as numpy array
            pixel_distance = np.linalg.norm(p2 - p1)  # Euclidean distance
            pixels_per_meter = pixel_distance / line_length

            request.session['pixels_per_meter'] = pixels_per_meter #Almacena en sesión
            # --- Store Camera Data ---
            camera_uuid, existing_camera_data = get_camera_by_ip(ip_port)
            if camera_uuid: # Check for duplicate
                return JsonResponse({'success': False, 'message': 'Ya existe una cámara con esta IP/URL.'})

            camera_uuid = str(uuid.uuid4())
            camera_data = {
                "ip_url": ip_port,
                "label": camera_name,
                "px_m": pixels_per_meter,  # Use the calculated value
                "status": "calibrated",
                "uuid": camera_uuid,
                "username": username,
                "password": password,
                "protocol": protocol,
                "full_url": full_url
            }
            save_camera_data(ambiente, piso, camera_uuid, camera_data)
            create_log_entry(request, 'Calibrar Cámara', f'Finalizó calibración (línea) en cámara: {ip_port}')
            return JsonResponse({'success': True, 'pixels_per_meter': pixels_per_meter})

    return render(request, 'calibrar_camara.html', context)

def store_line_points(request):
    """Stores the line points in the session."""
    if request.method == 'POST' and request.headers.get('X-Requested-With') == 'XMLHttpRequest':
        try:
            data = json.loads(request.body)  # Correctly load JSON data
            points = data.get('points')      # Access the 'points' key
            if points and len(points) == 2:   # Validate the points
                request.session['line_points'] = points
                return JsonResponse({'success': True})
            else:
                return JsonResponse({'success': False, 'message': 'Se requieren exactamente dos puntos para una línea.'})
        except json.JSONDecodeError:
            return JsonResponse({'success': False, 'message': 'Datos JSON inválidos.'})
    return JsonResponse({'success': False, 'message':"Invalid request"})



#------------------------------------------------------------------------------------------------




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

def obtener_datos_monitoreo():
    ambientes_ref = db.reference('/ambientes')
    ambientes_data = ambientes_ref.get()
    datos_monitoreo = []
    resumen_ambientes = defaultdict(lambda: {'personas_detectadas': 0, 'max_personas': 0, 'asientos_disponibles': 0, 'ultima_deteccion': 0})
    camaras_configuradas = []

    if isinstance(ambientes_data, dict):
        for ambiente_nombre, ambiente_data in ambientes_data.items():
            pisos = ambiente_data.get('piso', {})
            if isinstance(pisos, dict):
                for piso_nombre, piso_data in pisos.items():
                    camaras = piso_data.get('camara', {})
                    if isinstance(camaras, dict):
                        for cam_uuid, cam_data in camaras.items():
                            if cam_data.get('status') == 'configurated':
                                last_detection_time = cam_data.get('last_detection_time', 0)
                                current_time = time.time()
                                time_diff = current_time - last_detection_time  # Calcula SIEMPRE la diferencia

                                connection_state = cam_data.get('connection_state', 'stable')
                                connection_attempts = cam_data.get('connection_attempts', 0)
                                cam_label = cam_data.get('label', 'Sin etiqueta')
                                mesas = cam_data.get('mesas', [])

                                camaras_configuradas.append({
                                    "ambiente": ambiente_nombre,
                                    "piso": piso_nombre,
                                    "camara": cam_label,
                                    "uuid": cam_uuid,
                                    "connection_state": connection_state,
                                    "connection_attempts": connection_attempts,
                                    "antiguedad": int(time_diff),  # Siempre incluye la antigüedad
                                })

                                # --- Filtro de tiempo SOLO para datos_monitoreo y resumen_ambientes ---
                                if time_diff <= 150:
                                    if isinstance(mesas, list):
                                        for mesa in mesas:
                                            personas_detectadas = mesa.get("personas_Detectadas", 0)
                                            max_personas = mesa.get("max_personas", 0)
                                            asientos_disponibles = max_personas - personas_detectadas

                                            mesa_info = {
                                                "ambiente": ambiente_nombre,
                                                "piso": piso_nombre,
                                                "camara": cam_label,
                                                "mesa_id": mesa.get("id", "N/A"),
                                                "max_personas": max_personas,
                                                "personas_detectadas": personas_detectadas,
                                                "asientos_disponibles": asientos_disponibles,
                                                "antiguedad": int(time_diff), # Agregado para la tabla de detalle
                                            }
                                            datos_monitoreo.append(mesa_info)

                                            resumen_ambientes[ambiente_nombre]['personas_detectadas'] += personas_detectadas
                                            resumen_ambientes[ambiente_nombre]['max_personas'] += max_personas
                                            resumen_ambientes[ambiente_nombre]['asientos_disponibles'] += asientos_disponibles
                                            # Usamos la mayor ultima deteccion entre las mesas de la camara.
                                            resumen_ambientes[ambiente_nombre]['ultima_deteccion'] = max(
                                                resumen_ambientes[ambiente_nombre]['ultima_deteccion'],
                                                last_detection_time
                                            )

    # Calcula la antigüedad para el resumen de ambientes
    for ambiente, datos in resumen_ambientes.items():
        if datos['ultima_deteccion'] > 0:
            # Calcula la diferencia, ahora correctamente
            datos['antiguedad'] = int(time.time() - datos['ultima_deteccion'])
        else:
            datos['antiguedad'] = None  # O usa None si no hay detecciones

    return datos_monitoreo, dict(resumen_ambientes), camaras_configuradas



@never_cache
@login_required
def monitoring_view(request):
    datos_monitoreo, resumen_ambientes, camaras_configuradas = obtener_datos_monitoreo()
    return render(request, 'monitoreo.html', {
        'datos_monitoreo': datos_monitoreo,
        'resumen_ambientes': resumen_ambientes,
        'camaras_configuradas': camaras_configuradas
    })


@login_required
@never_cache
def get_monitoreo_data(request):
    datos_monitoreo, resumen_ambientes, camaras_configuradas = obtener_datos_monitoreo()

    # Filtra resumen_ambientes para incluir solo aquellos con detecciones recientes
    resumen_ambientes_filtrado = {}
    for ambiente, datos in resumen_ambientes.items():
        #  Filtra *aquí*, justo antes de enviar los datos
        if datos['antiguedad'] is not None and datos['antiguedad'] <= 150:
            resumen_ambientes_filtrado[ambiente] = datos

    data = {
        'datos_monitoreo': datos_monitoreo,
        'resumen_ambientes': resumen_ambientes_filtrado,  # Usa el resumen filtrado
        'camaras_configuradas': camaras_configuradas
    }
    return JsonResponse(data)

@login_required
def eliminar_camaras(request):
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            cameras_data = data.get('cameras', [])  # Cambiado de 'camera_uuids' a 'cameras'

            if not cameras_data:
                return JsonResponse({'success': False, 'error': 'No se proporcionaron datos de cámaras para eliminar.'})

            for camera in cameras_data:
                camera_uuid = camera.get('uuid')
                ambiente = camera.get('ambiente')
                piso = camera.get('piso')
                camera_name = camera.get('name')

                delete_camera_from_firebase(camera_uuid)
                create_log_entry(request, 'Eliminar cámara', f'Cámara eliminada: {camera_name}, Ambiente: {ambiente}, Piso: {piso}, UUID: {camera_uuid}')

            return JsonResponse({'success': True})

        except json.JSONDecodeError:
            return JsonResponse({'success': False, 'error': 'Error al decodificar JSON.'})
        except Exception as e:
            return JsonResponse({'success': False, 'error': f'Error al eliminar las cámaras: {str(e)}'})

    return JsonResponse({'success': False, 'error': 'Método no permitido'})


##Seccion Logs:

@login_required
@role_required(allowed_roles=['admin'])  # Asegúrate de que solo los administradores puedan acceder
def admin_logs(request):
    logs = LogEntry.objects.all().order_by('-timestamp')
    return render(request, 'admin_logs.html', {'logs': logs})