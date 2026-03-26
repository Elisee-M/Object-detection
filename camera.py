from ultralytics import YOLO
import cv2

# Load YOLO model
model = YOLO("yolov8n.pt")

# Open webcam (0 = default camera)
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("❌ Cannot access camera")
    exit()

while True:
    ret, frame = cap.read()
    
    if not ret:
        print("❌ Failed to grab frame")
        break

    # Run YOLO on the frame
    results = model(frame)

    # Draw detections on frame
    annotated_frame = results[0].plot()

    # Show frame
    cv2.imshow("YOLO Live Detection", annotated_frame)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()