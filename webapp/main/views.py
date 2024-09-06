from django.shortcuts import render, redirect
import cv2
import logging
from django.http import StreamingHttpResponse, HttpResponseBadRequest,JsonResponse
import urllib.request
from django.shortcuts import render
import numpy as np

logger = logging.getLogger(__name__)

def login_view(request):
    if request.method == 'POST':
        # Aquí iría la lógica de autenticación en el futuro
        return redirect('monitoreo')
    return render(request, 'login.html')

def monitoring_view(request):
    return render(request, 'monitoreo.html')

def configuracion_view(request):
    return render(request, 'configuracion.html')


from django.shortcuts import render, redirect
import cv2
import logging
from django.http import StreamingHttpResponse, HttpResponseBadRequest, JsonResponse
import numpy as np

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
                    update_camera_data(
                        request.session.get('ambiente'),
                        request.session.get('piso'),
                        camera_data
                    )
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

            if len(pixels_per_meter_near) < 30 or len(pixels_per_meter_far) < 30:
                return JsonResponse({'success': False, 'message': 'Se necesitan al menos 30 capturas para cada distancia'})

            pixels_per_meter = np.mean(pixels_per_meter_near[:30] + pixels_per_meter_far[:30])
            
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


#EOF
#------------------------------------------------------------------------------------------------------------------------------------------#

####SECCION BASE DE DATOS JSON : CAMBIAR A BASE DE DATOS FIREBASE:
import json
import os

JSON_FILE_PATH = 'database.json'

def load_json_data():
    if os.path.exists(JSON_FILE_PATH):
        with open(JSON_FILE_PATH, 'r') as file:
            return json.load(file)
    return {"ambientes": []}

def save_json_data(data):
    with open(JSON_FILE_PATH, 'w') as file:
        json.dump(data, file, indent=2)

def update_camera_data(ambiente, piso, camera_data):
    data = load_json_data()
    for amb in data["ambientes"]:
        if amb["nombreAmbiente"] == ambiente:
            if piso not in amb["piso"]:
                amb["piso"][piso] = {"camara": []}
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












