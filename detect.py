from ultralytics import YOLO

model = YOLO("yolov8n.pt")
model("validate/fruits.jpg", show=True, save=True)
