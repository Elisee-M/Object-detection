from ultralytics import YOLO

model = YOLO("yolov8n.pt")
model("animals.png", show=True, save=True)
