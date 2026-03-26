from ultralytics import YOLO

model = YOLO("yolov8n.pt")
model("validate/animals.png", show=True, save=True)
