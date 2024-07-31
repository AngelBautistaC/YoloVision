import cv2

from ultralytics import YOLO, solutions


#Con el uso de la funcion Distance Calculation, se puede calcular la distancia entre el centroide de dos objetos.
model = YOLO('yolov8l.pt')
names = model.model.names

cap = cv2.VideoCapture("./16.00_16.30.mkv")
assert cap.isOpened(), "Error reading video file"
w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))

# Video writer
video_writer = cv2.VideoWriter("distance_calculation.avi", cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

# Init distance-calculation obj
dist_obj = solutions.DistanceCalculation(names=names,pixels_per_meter=340, view_img=True)

while cap.isOpened():
    success, im0 = cap.read()
    if not success:
        print("Video frame is empty or video processing has been successfully completed.")
        break

    tracks = model.track(im0, persist=True, show=False)
    im0 = dist_obj.start_process(im0, tracks)
    video_writer.write(im0)

cap.release()
video_writer.release()
cv2.destroyAllWindows()