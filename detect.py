from ultralytics import YOLO

model = YOLO("yolov8n.pt")
model("video.mp4", show=True, save=True)
