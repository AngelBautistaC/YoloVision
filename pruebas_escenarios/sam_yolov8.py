from ultralytics import YOLO

# Load a segmentation model
model = YOLO('yolov8s-seg.pt')

# Predict with the model
results = model.predict('C:/Users/a2019/Desktop/Tesiscode/pruebas_escenarios/mesa.jpg')
results[0].show()