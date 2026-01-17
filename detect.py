from ultralytics import YOLO

model = YOLO("yolov8n.pt")
model("stress.mp4", show=True, save=True)
