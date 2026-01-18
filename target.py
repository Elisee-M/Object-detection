from ultralytics import YOLO

# Load YOLO model
model = YOLO("yolov8n.pt")

# Image to analyze
image_path = "validate/fruits.png"   # change to your image

# Object you want to search for
target_object = "dog"       # example: person, car, cat, bottle

# Run detection
results = model(image_path)

found = False

for r in results:
    for cls in r.boxes.cls:
        class_name = model.names[int(cls)]
        if class_name == target_object:
            found = True
            break

if found:
    print(f"✅ '{target_object}' found in the image!")
else:
    print(f"❌ '{target_object}' not found.")
