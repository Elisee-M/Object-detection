from ultralytics import YOLO

model = YOLO("yolov8n.pt")
model("bus.jpg", show=True, save=True)
