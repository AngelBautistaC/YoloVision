import cv2
import numpy as np
from ultralytics import YOLO
from ultralytics import YOLOWorld
import cvzone

# Este script realiza la detección de objetos de naturaleza tabla, y se busca implementar la medición del objeto.

# Configuración de la fuente de video
use_ip_camera = False
use_yolo = True
use_yoloworld = False

# Configuración de la fuente de video
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

if use_yolo:
    if use_yoloworld:
        model = YOLOWorld('yolov8m-worldv2.pt')
        table_keywords = [
            "desk", "table", "dining table", "library table", "office table",
            "meeting table", "workbench", "coffee table", "conference table",
            "study table", "kitchen table", "side table", "end table",
            "craft table", "picnic table", "bar table", "console table",
            "dressing table", "folding table", "round table", "square table",
            "rectangular table"
        ]
        model.set_classes(table_keywords)
    else:
        model = YOLO('yolov8l-oiv7.pt')

cap = cv2.VideoCapture(video_source)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print(f"Dimensiones del stream: {width}x{height}")

pixels_per_meter = None
captured_image = None
detections = []
dragging = False
current_point = None
moving_box = None
last_pos = None
detection_active = True
pixels_per_meter_buffer = []


def mouse_callback(event, x, y, flags, param):
    global dragging, current_point, moving_box, last_pos

    if event == cv2.EVENT_LBUTTONDOWN:
        for i, detection in enumerate(detections):
            for j, point in enumerate(detection['points']):
                if np.sqrt((x - point[0])**2 + (y - point[1])**2) < 10:
                    current_point = (i, j)
                    dragging = True
                    return
            
            if cv2.pointPolygonTest(np.array(detection['points']), (x, y), False) >= 0:
                moving_box = i
                last_pos = (x, y)
                return

    elif event == cv2.EVENT_MOUSEMOVE:
        if dragging and current_point is not None:
            i, j = current_point
            detections[i]['points'][j] = (x, y)
        elif moving_box is not None and last_pos is not None:
            dx = x - last_pos[0]
            dy = y - last_pos[1]
            detections[moving_box]['points'] = [(p[0] + dx, p[1] + dy) for p in detections[moving_box]['points']]
            last_pos = (x, y)

    elif event == cv2.EVENT_LBUTTONUP:
        dragging = False
        moving_box = None
        current_point = None
        last_pos = None


def calculate_area(pts):
    pts = np.array(pts, dtype=np.float32)
    return cv2.contourArea(pts)

cv2.namedWindow("Table Detection and Measurement")
cv2.setMouseCallback("Table Detection and Measurement", mouse_callback)


while True:
    if detection_active:
        ret, frame = cap.read()
        if not ret:
            print("Error al capturar el frame.")
            break

        corners, ids, rejected = detector.detectMarkers(frame)
        if ids is not None:
            cv2.aruco.drawDetectedMarkers(frame, corners, ids)
            for i in range(len(ids)):
                marker_width_px = np.linalg.norm(corners[i][0][1] - corners[i][0][0])
                pixels_per_meter_buffer.append(marker_width_px / marker_size)
            
            print(f"Datos de pixels recolectados: {len(pixels_per_meter_buffer)}")
            cv2.putText(frame, f"Datos: {len(pixels_per_meter_buffer)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        if use_yolo and pixels_per_meter is not None:
            results = model(frame)
            new_detections = []
            for result in results:
                boxes = result.boxes
                for box in boxes:
                    cls = int(box.cls[0])
                    conf = float(box.conf[0])
                    
                    if use_yoloworld or model.names[cls] in ['Table', 'Kitchen & dining room table', 'Coffee table']:
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        
                        new_detections.append({
                            'points': [(x1, y1), (x2, y1), (x2, y2), (x1, y2)],
                            'label': f"Mesa {len(new_detections) + 1}"
                        })
            
            if new_detections:
                captured_image = frame.copy()
                detections = new_detections
                detection_active = False  # Detener la detección

        cv2.imshow("Table Detection and Measurement", frame)
    
    if captured_image is not None:
        display_image = captured_image.copy()
        
        for i, detection in enumerate(detections):
            cv2.polylines(display_image, [np.array(detection['points'], dtype=np.int32)], True, (0, 255, 0), 2)
            for pt in detection['points']:
                cv2.circle(display_image, (int(pt[0]), int(pt[1])), 5, (0, 0, 255), -1)
            
            area_pixels = calculate_area(detection['points'])
            area_meters = area_pixels / (pixels_per_meter ** 2)
            
            info_text = f"{detection['label']}: {area_meters:.2f} m²"
            text_size = cv2.getTextSize(info_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
            text_x = int(detection['points'][0][0])
            text_y = int(detection['points'][0][1]) - 10
            cv2.rectangle(display_image, (text_x, text_y - text_size[1] - 10), (text_x + text_size[0], text_y), (0, 0, 0), -1)
            cv2.putText(display_image, info_text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        cv2.putText(display_image, f"Pixels per meter: {pixels_per_meter:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.imshow("Table Detection and Measurement", display_image)
    
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('r'):
        captured_image = None
        detections = []
        detection_active = True
    elif key == ord('c') and len(pixels_per_meter_buffer) > 0:
        pixels_per_meter = np.mean(pixels_per_meter_buffer)
        print(f"Valor promedio de pixels_per_meter calculado: {pixels_per_meter:.2f}")
        pixels_per_meter_buffer = []

cap.release()
cv2.destroyAllWindows()