from ultralytics import YOLO

model = YOLO("yolov8n.pt")
model("image.jpg", show=True)
