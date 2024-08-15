import cv2
import numpy as np
import os

# Definir las dimensiones del tablero de ajedrez
CHECKERBOARD = (6, 9)
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# Crear vectores para almacenar puntos 3D y 2D
objpoints = []
imgpoints = []

# Definir coordenadas del mundo real para puntos 3D
objp = np.zeros((1, CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
objp[0,:,:2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)

# Inicializar la cámara
cap = cv2.VideoCapture(0)

captures = []
calibrated = False

print("Presione 'c' para capturar una imagen, 'q' para salir, o 'k' para calibrar (después de al menos 10 capturas)")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error al capturar la imagen")
        break

    cv2.imshow('Calibración de cámara', frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('c'):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, 
                                                 cv2.CALIB_CB_ADAPTIVE_THRESH + 
                                                 cv2.CALIB_CB_FAST_CHECK + 
                                                 cv2.CALIB_CB_NORMALIZE_IMAGE)
        if ret:
            objpoints.append(objp)
            corners2 = cv2.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)
            imgpoints.append(corners2)
            captures.append(frame)
            print(f"Captura {len(captures)} guardada. Total: {len(captures)}")
        else:
            print("No se detectó el patrón del tablero de ajedrez")

    elif key == ord('k') and len(captures) >= 10:
        print("Calibrando la cámara...")
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
        print("Matriz de la cámara:")
        print(mtx)
        print("Coeficientes de distorsión:")
        print(dist)
        calibrated = True
        break

    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

if calibrated:
    print("Calibración completada y matriz de la cámara obtenida.")
else:
    print("Calibración cancelada.")
