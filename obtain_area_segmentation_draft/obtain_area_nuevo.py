import cv2
import numpy as np
from ultralytics import YOLO
import math
import tracker  # Importamos el módulo tracker

#Para la detección de personas en el area, se está considerando el centroide de la imagen segementada obtenida de una persona
# es decir, su polígono de detección.

# Configuración de la fuente de video
use_ip_camera = False
video_source = 'http://192.168.31.128:8080/' if use_ip_camera else 0

# ArUco configuration
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_50)
aruco_params = cv2.aruco.DetectorParameters()
detector = cv2.aruco.ArucoDetector(aruco_dict, aruco_params)

# Camera calibration parameters
camera_matrix = np.array([
    [1.42981228e+03, 0.00000000e+00, 6.79469267e+02],
    [0.00000000e+00, 1.43227361e+03, 3.52205086e+02],
    [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]
], dtype=np.float32)
dist_coeffs = np.array([0, 0, 0, 0, 0], dtype=np.float32)

marker_size = 0.15  # 15 cm

# Cargar el modelo YOLOv8 de segmentación
model = YOLO('yolov8m-seg.pt')

cap = cv2.VideoCapture(video_source)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print(f"Dimensiones del stream: {width}x{height}")

pixels_per_meter = None
captured_image = None
detections = []
pixels_per_meter_buffer_near = []
pixels_per_meter_buffer_far = []
detection_active = True
calibration_stage = 0  # 0: sin empezar, 1: cerca, 2: lejos, 3: completado
current_polygon = None
expanded_polygon = None
tracking_active = False
detection_paused = False  # Nueva variable


personas_por_perimetro = 1.15 # Cuantas personas entran en un segmento de X metros? 1 persona por cada 1.15 metros
distancia_persona = 0.3  # Cuanto es el margen desde la mesa a la personas para ser detectadas? 30 cm
personas=0
dragging = False
current_point = None
moving_box = False
last_pos = None

# Variables para el tracking
track_history = None
counting_regions = None
max_persons = 0

def simplify_polygon(polygon, target_points=6):
    epsilon = 0.01 * cv2.arcLength(polygon, True)
    approx = cv2.approxPolyDP(polygon, epsilon, True)
    while len(approx) > target_points:
        epsilon *= 1.1
        approx = cv2.approxPolyDP(polygon, epsilon, True)
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

def expand_polygon(polygon, distance):
    poly_np = np.array(polygon, dtype=np.float32)
    M = cv2.moments(poly_np)
    cx = int(M['m10'] / M['m00'])
    cy = int(M['m01'] / M['m00'])
    center = np.array([cx, cy])
    expanded = []
    for point in poly_np:
        vector = point - center
        vector_norm = vector / np.linalg.norm(vector)
        expanded_point = point + vector_norm * distance * pixels_per_meter
        expanded.append(expanded_point)
    return expanded

def mouse_callback(event, x, y, flags, param):
    global dragging, current_point, moving_box, last_pos, current_polygon

    if current_polygon is None:
        return

    if event == cv2.EVENT_LBUTTONDOWN:
        for i, point in enumerate(current_polygon):
            if np.sqrt((x - point[0])**2 + (y - point[1])**2) < 10:
                current_point = i
                dragging = True
                return
        
        if cv2.pointPolygonTest(np.array(current_polygon), (x, y), False) >= 0:
            moving_box = True
            last_pos = (x, y)
            return

    elif event == cv2.EVENT_MOUSEMOVE:
        if dragging and current_point is not None:
            current_polygon[current_point] = (x, y)
        elif moving_box and last_pos is not None:
            dx = x - last_pos[0]
            dy = y - last_pos[1]
            current_polygon = [(p[0] + dx, p[1] + dy) for p in current_polygon]
            last_pos = (x, y)

    elif event == cv2.EVENT_LBUTTONUP:
        dragging = False
        moving_box = False
        current_point = None
        last_pos = None


def reset_all():
    global pixels_per_meter, captured_image, detections, pixels_per_meter_buffer_near, pixels_per_meter_buffer_far, detection_active, calibration_stage, current_polygon, detection_paused
    pixels_per_meter = None
    captured_image = None
    detections = []
    pixels_per_meter_buffer_near = []
    pixels_per_meter_buffer_far = []
    detection_active = True
    calibration_stage = 0
    current_polygon = None
    detection_paused = False  # Reiniciar también esta variable


cv2.namedWindow("Table Detection and Measurement")
cv2.setMouseCallback("Table Detection and Measurement", mouse_callback)



while True:
    ret, frame = cap.read()
    if not ret:
        print("Error al capturar el frame.")
        break

    if detection_active:
        if calibration_stage < 3:
            corners, ids, rejected = detector.detectMarkers(frame)
            if ids is not None:
                cv2.aruco.drawDetectedMarkers(frame, corners, ids)
                for i in range(len(ids)):
                    marker_width_px = np.linalg.norm(corners[i][0][1] - corners[i][0][0])
                    if calibration_stage == 1 and len(pixels_per_meter_buffer_near) < 30:
                        pixels_per_meter_buffer_near.append(marker_width_px / marker_size)
                    elif calibration_stage == 2 and len(pixels_per_meter_buffer_far) < 30:
                        pixels_per_meter_buffer_far.append(marker_width_px / marker_size)
                
            if calibration_stage == 0:
                cv2.putText(frame, "Coloque ArUco en el lado cercano y presione 'c'", (10, height - 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            elif calibration_stage == 1:
                cv2.putText(frame, f"Datos cercanos: {len(pixels_per_meter_buffer_near)}/30", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                if len(pixels_per_meter_buffer_near) >= 30:
                    cv2.putText(frame, "Presione 'f' para iniciar medicion lejana", (10, height - 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            elif calibration_stage == 2:
                cv2.putText(frame, f"Datos lejanos: {len(pixels_per_meter_buffer_far)}/30", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                if len(pixels_per_meter_buffer_far) >= 30:
                    cv2.putText(frame, "Presione 'p' para calcular pixels por metro", (10, height - 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        if pixels_per_meter is not None:
            if not detection_paused:
                results = model(frame, classes=[60], conf=0.1)
                detections = []

                for result in results:
                    masks = result.masks
                    if masks is not None:
                        for i, mask in enumerate(masks):
                            mask_array = mask.xy[0].astype(int)
                            polygon = cv2.convexHull(mask_array)
                            simplified_polygon = simplify_polygon(polygon)
                            if len(simplified_polygon) > 2:
                                detections.append({
                                    'points': simplified_polygon.tolist(),
                                    'label': f"Tabla {i+1}"
                                })
                                cv2.polylines(frame, [simplified_polygon], isClosed=True, color=(0, 255, 0), thickness=2)
                                M = cv2.moments(simplified_polygon)
                                if M["m00"] != 0:
                                    cX = int(M["m10"] / M["m00"])
                                    cY = int(M["m01"] / M["m00"])
                                    cv2.putText(frame, f"Tabla {i+1}", (cX, cY), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                        
                        if detections:
                            detection_paused = True

            if detection_paused:
                cv2.putText(frame, "Presione 'u' para usar este poligono o 'c' para continuar buscando", (10, height - 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                for detection in detections:
                    cv2.polylines(frame, [np.array(detection['points'], dtype=np.int32)], isClosed=True, color=(0, 255, 0), thickness=2)
            else:
                cv2.putText(frame, "Buscando mesa...", (10, height - 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)


        cv2.imshow("Table Detection and Measurement", frame)  
        #cv2.imshow("Table Detection and Measurement", frame)

    elif not tracking_active:
        if captured_image is not None:
            display_image = captured_image.copy()
            
            if current_polygon is not None:
                cv2.polylines(display_image, [np.array(current_polygon, dtype=np.int32)], True, (0, 255, 0), 2)
                for pt in current_polygon:
                    cv2.circle(display_image, (int(pt[0]), int(pt[1])), 5, (0, 0, 255), -1)
                
                area_pixels = cv2.contourArea(np.array(current_polygon))
                area_meters = area_pixels / (pixels_per_meter ** 2)
                
                perimeter_pixels = cv2.arcLength(np.array(current_polygon), True)
                perimeter_meters = perimeter_pixels / pixels_per_meter
                
                personas = math.floor(perimeter_meters / personas_por_perimetro)
                
                expanded_polygon = expand_polygon(current_polygon, distancia_persona)
                cv2.polylines(display_image, [np.array(expanded_polygon, dtype=np.int32)], True, (255, 0, 0), 2)
                
                info_text = f"Area: {area_meters:.2f} m^2 | Perimetro: {perimeter_meters:.2f} m | Personas max: {personas}"
                cv2.putText(display_image, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                cv2.putText(display_image, "Presione 't' para iniciar el tracking", (10, height - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            
            cv2.imshow("Table Detection and Measurement", display_image)

    else:  # tracking_active
        # Realizar el tracking y conteo
        frame_with_tracking = tracker.track_and_count(frame, model, track_history, counting_regions)
        
        # Dibujar el polígono de la mesa y el expandido
        cv2.polylines(frame_with_tracking, [np.array(current_polygon, dtype=np.int32)], True, (0, 255, 0), 2)
        cv2.polylines(frame_with_tracking, [np.array(expanded_polygon, dtype=np.int32)], True, (255, 0, 0), 2)
        
        # Mostrar información en pantalla
        info_text = f"Area: {area_meters:.2f} m^2 | Perimetro: {perimeter_meters:.2f} m | Personas max: {personas}"
        cv2.putText(frame_with_tracking, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        cv2.imshow("Table Detection and Measurement", frame_with_tracking)

    # Manejo de teclas
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('c') and calibration_stage == 0:
        calibration_stage = 1
        pixels_per_meter_buffer_near = []
    elif key == ord('f') and calibration_stage == 1 and len(pixels_per_meter_buffer_near) >= 30:
        calibration_stage = 2
        pixels_per_meter_buffer_far = []
    elif key == ord('p') and calibration_stage == 2 and len(pixels_per_meter_buffer_far) >= 30:
        pixels_per_meter = np.mean(pixels_per_meter_buffer_near[:30] + pixels_per_meter_buffer_far[:30])
        print(f"Valor promedio de pixels_per_meter calculado: {pixels_per_meter:.2f}")
        calibration_stage = 3
    elif key == ord('r') and pixels_per_meter is not None and detections:
        captured_image = frame.copy()
        current_polygon = detections[0]['points']
        detection_active = False
    elif key == ord('t') and not tracking_active and current_polygon is not None:
        tracking_active = True
        track_history = tracker.initialize_tracking()
        counting_regions = tracker.initialize_counting_regions(expanded_polygon, personas)
        model = YOLO('yolov8s-seg.pt')
    elif key == ord('n'):
        reset_all()
        tracking_active = False
    elif key == ord('u') and detection_paused:
        # Usar el polígono actual
        current_polygon = detections[0]['points']
        detection_active = False
    elif key == ord('c') and detection_paused:
        # Continuar buscando
        detection_paused = False
        detections = []

# Cerrar la captura de video y las ventanas
cap.release()
cv2.destroyAllWindows()