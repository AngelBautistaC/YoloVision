from ultralytics import YOLOWorld

# Initialize the model with pre-trained weights
model = YOLOWorld('yolov8m-worldv2.pt')

table_keywords = [
    "desk",
    "table",
    "dining table",
    "library table",
    "office table",
    "meeting table",
    "workbench",
    "coffee table",
    "conference table",
    "study table",
    "kitchen table",
    "side table",
    "end table",
    "craft table",
    "picnic table",
    "bar table",
    "console table",
    "dressing table",
    "folding table",
    "round table",
    "square table",
    "rectangular table"
]


# Set what you'd like to find in your image
model.set_classes(table_keywords)

# Run object detection for your custom classes on your image
results = model.predict("C:/Users/a2019/Desktop/Tesiscode/obtain_area_yoloworld/WIN_20240730_18_15_44_Pro.jpg", conf=0.1, iou=0.5)
# Display the results
results[0].show()