import cv2
import numpy as np
from ultralytics import YOLO

# Cargar el modelo YOLOv8 de segmentación
model = YOLO('yolov8x-seg.pt')

# Iniciar la captura de video desde la cámara web
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

while True:
    # Capturar frame por frame
    ret, frame = cap.read()
    if not ret:
        break

    # Realizar la detección y segmentación con YOLO
    results = model(frame, classes=[60])

    # Dibujar los resultados en el frame
    for result in results:
        masks = result.masks
        if masks is not None:
            for mask in masks:
                # Extraer y simplificar el polígono de la máscara
                mask_array = mask.xy[0].astype(int)
                polygon = cv2.convexHull(mask_array)

                # Dibujar el polígono
                cv2.polylines(frame, [polygon], isClosed=True, color=(0, 255, 0), thickness=2)

                # Calcular el centro del polígono para colocar la etiqueta
                M = cv2.moments(polygon)
                if M["m00"] != 0:
                    cX = int(M["m10"] / M["m00"])
                    cY = int(M["m01"] / M["m00"])
                    # Añadir etiqueta
                    cv2.putText(frame, "Clase 60", (cX, cY), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Mostrar el resultado
    cv2.imshow("Segmentación YOLO - Clase 60", frame)

    # Salir si se presiona 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberar la captura y cerrar las ventanas
cap.release()
cv2.destroyAllWindows()