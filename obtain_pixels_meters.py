import cv2
import numpy as np
from ultralytics import YOLO
import cvzone

# Selección de la fuente de video
use_ip_camera = False  # Cambie a False para usar la cámara integrada

# Configuración de la fuente de video
if use_ip_camera:
    video_source = 'http://192.168.31.128:8080/'  # URL de la cámara IP
else:
    video_source = 0  # 0 para la cámara integrada

# ArUco configuration
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_50)
aruco_params = cv2.aruco.DetectorParameters()
detector = cv2.aruco.ArucoDetector(aruco_dict, aruco_params)

# Camera calibration parameters (use your own if available)
camera_matrix = np.array([
    [1.42981228e+03, 0.00000000e+00, 6.79469267e+02],
    [0.00000000e+00, 1.43227361e+03, 3.52205086e+02],
    [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]
], dtype=np.float32)
dist_coeffs = np.array([0, 0, 0, 0, 0], dtype=np.float32)

# ArUco marker size in meters
marker_size = 0.15  # 15 cm

# Load YOLOv8 model
model = YOLO('yolov8s-seg.pt')

# Capturar el video
cap = cv2.VideoCapture(video_source)

# Obtener las dimensiones del stream
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print(f"Dimensiones del stream: {width}x{height}")

pixels_per_meter = None

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error al capturar el frame.")
        break

    # ArUco detection
    corners, ids, rejected = detector.detectMarkers(frame)
    if ids is not None:
        rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(corners, marker_size, camera_matrix, dist_coeffs)
        for i in range(len(ids)):
            cv2.aruco.drawDetectedMarkers(frame, corners, ids)
            
            # Calculate pixels per meter
            marker_width_px = np.linalg.norm(corners[i][0][1] - corners[i][0][0])
            pixels_per_meter = marker_width_px / marker_size
            cv2.putText(frame, f"Pixels per meter: {pixels_per_meter:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    # YOLOv8 detection (focusing on tables)
    results = model(frame, classes=[60,67])  # 60 is the class ID for tables in COCO dataset

    for result in results:
        masks = result.masks
        boxes = result.boxes

        if masks is not None and boxes is not None:
            for mask, box in zip(masks, boxes):
                # Extract polygon from mask
                mask_array = mask.xy[0].astype(int)
                polygon = cv2.convexHull(mask_array)

                # Draw polygon
                cv2.polylines(frame, [polygon], isClosed=True, color=(0, 255, 0), thickness=2)

                # Calculate table area
                if pixels_per_meter is not None:
                    area_pixels = cv2.contourArea(polygon)
                    area_meters = area_pixels / (pixels_per_meter ** 2)
                    
                    # Display area
                    cvzone.putTextRect(frame, f"Table Area: {area_meters:.2f} m^2", (10, 70), scale=1, thickness=2, colorR=(0,255,0), colorT=(255,255,255), offset=10)

    cv2.imshow("Table Detection and Measurement", frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()