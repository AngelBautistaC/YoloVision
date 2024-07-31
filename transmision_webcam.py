import cv2
import numpy as np

#Este script captura la transmisión de un stream de video
# URL de la transmisión
url = 'http://192.168.31.128:8080/'

# Capturar el video
cap = cv2.VideoCapture(url)

while True:
    # Leer un frame
    ret, frame = cap.read()
    
    if not ret:
        print("Error al capturar el frame.")
        break
    
    # Mostrar el frame
    cv2.imshow('Transmisión', frame)
    
    # Salir si se presiona 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberar recursos
cap.release()
cv2.destroyAllWindows()