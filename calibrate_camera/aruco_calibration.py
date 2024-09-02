import cv2
import numpy as np

# Configuración de la cámara
video_source = 0
cap = cv2.VideoCapture(video_source)
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_50)
aruco_params = cv2.aruco.DetectorParameters()
detector = cv2.aruco.ArucoDetector(aruco_dict, aruco_params)

# Tamaño del marcador ArUco
marker_size = 0.15  # 15 cm

pixels_per_meter_buffer_near = []
pixels_per_meter_buffer_far = []
calibration_stage = 0  # 0: sin empezar, 1: cerca, 2: lejos, 3: completado
pixels_per_meter = None

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error al capturar el frame.")
        break

    corners, ids, _ = detector.detectMarkers(frame)
    if ids is not None:
        cv2.aruco.drawDetectedMarkers(frame, corners, ids)
        for i in range(len(ids)):
            marker_width_px = np.linalg.norm(corners[i][0][1] - corners[i][0][0])
            marker_perimeter_px = cv2.arcLength(corners[i][0], True)  # Perímetro del marcador en píxeles

            if calibration_stage == 1 and len(pixels_per_meter_buffer_near) < 30:
                pixels_per_meter_buffer_near.append(marker_width_px / marker_size)
            elif calibration_stage == 2 and len(pixels_per_meter_buffer_far) < 30:
                pixels_per_meter_buffer_far.append(marker_width_px / marker_size)

            if pixels_per_meter is not None:
                marker_perimeter_m = marker_perimeter_px / pixels_per_meter  # Convertir el perímetro a metros
                cv2.putText(frame, f"Perimetro ArUco: {marker_perimeter_m:.2f} m", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    if calibration_stage == 0:
        cv2.putText(frame, "Coloque ArUco en el lado cercano y presione 'c'", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    elif calibration_stage == 1:
        cv2.putText(frame, f"Datos cercanos: {len(pixels_per_meter_buffer_near)}/30", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        if len(pixels_per_meter_buffer_near) >= 30:
            cv2.putText(frame, "Presione 'f' para iniciar medicion lejana", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    elif calibration_stage == 2:
        cv2.putText(frame, f"Datos lejanos: {len(pixels_per_meter_buffer_far)}/30", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        if len(pixels_per_meter_buffer_far) >= 30:
            cv2.putText(frame, "Presione 'p' para calcular pixels por metro", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    cv2.imshow("Calibration", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('c') and calibration_stage == 0:
        calibration_stage = 1
    elif key == ord('f') and calibration_stage == 1 and len(pixels_per_meter_buffer_near) >= 30:
        calibration_stage = 2
    elif key == ord('p') and calibration_stage == 2 and len(pixels_per_meter_buffer_far) >= 30:
        pixels_per_meter = np.mean(pixels_per_meter_buffer_near[:30] + pixels_per_meter_buffer_far[:30])
        print(f"Valor promedio de pixels_per_meter calculado: {pixels_per_meter:.2f}")
        calibration_stage = 3

cap.release()
cv2.destroyAllWindows()
