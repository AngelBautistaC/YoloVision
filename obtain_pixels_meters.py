import cv2
import numpy as np
from ultralytics import YOLO
from ultralytics import YOLOWorld
import cvzone

#Este script realiza la detección de objetos de naturaleza tabla, y se busca implementar la medición del objeto.

# Configuración de la fuente de video
use_ip_camera = False
use_yolo = True
use_yoloworld = True

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
points = []
dragging = False
current_point = None
moving_box = False
last_pos = None
detection_active = True

def mouse_callback(event, x, y, flags, param):
    global points, dragging, current_point, moving_box, last_pos

    if event == cv2.EVENT_LBUTTONDOWN:
        for i, point in enumerate(points):
            if np.sqrt((x - point[0])**2 + (y - point[1])**2) < 10:
                current_point = i
                dragging = True
                return
        
        if points and cv2.pointPolygonTest(np.array(points), (x, y), False) >= 0:
            moving_box = True
            last_pos = (x, y)

    elif event == cv2.EVENT_MOUSEMOVE:
        if dragging and current_point is not None:
            points[current_point] = (x, y)
        elif moving_box:
            dx = x - last_pos[0]
            dy = y - last_pos[1]
            points = [(p[0] + dx, p[1] + dy) for p in points]
            last_pos = (x, y)

    elif event == cv2.EVENT_LBUTTONUP:
        dragging = False
        moving_box = False
        current_point = None
        last_pos = None

def calculate_area(pts):
    pts = np.array(pts, dtype=np.float32)
    return cv2.contourArea(pts)

def detect_table_surface(frame, aruco_corners):
    mask = np.ones(frame.shape[:2], dtype=np.uint8) * 255
    cv2.fillConvexPoly(mask, aruco_corners.astype(int), 0)

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    colors = []
    for corner in aruco_corners:
        x, y = corner.astype(int)
        for dx, dy in [(-10, 0), (10, 0), (0, -10), (0, 10)]:
            sample_x, sample_y = max(0, min(x + dx, frame.shape[1] - 1)), max(0, min(y + dy, frame.shape[0] - 1))
            colors.append(hsv[sample_y, sample_x])

    avg_color = np.mean(colors, axis=0)

    lower_bound = np.array([max(0, avg_color[0] - 30), max(0, avg_color[1] - 80), max(0, avg_color[2] - 80)])
    upper_bound = np.array([min(180, avg_color[0] + 30), min(255, avg_color[1] + 80), min(255, avg_color[2] + 80)])
    color_mask = cv2.inRange(hsv, lower_bound, upper_bound)

    final_mask = cv2.bitwise_and(color_mask, mask)

    kernel = np.ones((7,7), np.uint8)
    final_mask = cv2.morphologyEx(final_mask, cv2.MORPH_CLOSE, kernel, iterations=3)
    final_mask = cv2.morphologyEx(final_mask, cv2.MORPH_OPEN, kernel, iterations=2)

    edges = cv2.Canny(final_mask, 50, 150)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        epsilon = 0.01 * cv2.arcLength(largest_contour, True)
        approx = cv2.approxPolyDP(largest_contour, epsilon, True)
        return approx

    return None

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
            rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(corners, marker_size, camera_matrix, dist_coeffs)
            for i in range(len(ids)):
                cv2.aruco.drawDetectedMarkers(frame, corners, ids)
                marker_width_px = np.linalg.norm(corners[i][0][1] - corners[i][0][0])
                pixels_per_meter = marker_width_px / marker_size
                cv2.putText(frame, f"Pixels per meter: {pixels_per_meter:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        if use_yolo and pixels_per_meter is not None:
            results = model(frame)
            for result in results:
                boxes = result.boxes
                for box in boxes:
                    cls = int(box.cls[0])
                    conf = float(box.conf[0])
                    
                    if use_yoloworld or model.names[cls] in ['Table', 'Kitchen & dining room table', 'Coffee table']:
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        
                        if captured_image is None:
                            captured_image = frame.copy()
                            points = [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]
                            detection_active = False  # Detener la detección

        cv2.imshow("Table Detection and Measurement", frame)
    
    if captured_image is not None:
        display_image = captured_image.copy()
        
        # Dibujar solo los puntos y la caja que estamos editando
        cv2.polylines(display_image, [np.array(points, dtype=np.int32)], True, (0, 255, 0), 2)
        for pt in points:
            cv2.circle(display_image, (int(pt[0]), int(pt[1])), 5, (0, 0, 255), -1)
        
        area_pixels = calculate_area(points)
        area_meters = area_pixels / (pixels_per_meter ** 2)
        
        info_text = f"Area: {area_meters:.2f} m²"
        cv2.rectangle(display_image, (0, 0), (display_image.shape[1], 40), (0, 0, 0), -1)
        cv2.putText(display_image, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        cv2.imshow("Table Detection and Measurement", display_image)
    
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('r'):
        captured_image = None
        points = []
        detection_active = True

cap.release()
cv2.destroyAllWindows()