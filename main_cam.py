"Das ist inlet erkennung mit yolo local ohne api über webcam"

from ultralytics import YOLO
import cv2
import time
# Load the YOLO model
model = YOLO("./runs/detect/train/weights/best.pt")

# Initialize webcam
cap = cv2.VideoCapture(0)

# Check if the webcam is opened correctly
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

# Variables to calculate FPS
prev_time = 0

while True:
    # Capture frame-by-frame from webcam
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to capture image.")
        break
    
    # Measure the current time
    curr_time = time.time()
    fps = 1 / (curr_time - prev_time)
    prev_time = curr_time

    # Run predictions on the frame
    results = model.predict(frame,conf=0.4)
    
    # Process each detected object in the results
    for result in results:
        for box in result.boxes:
            # Get the bounding box coordinates
            x1, y1, x2, y2 = box.xyxy[0]  # Get coordinates of the first detected object
            confidence = box.conf[0]  # Confidence score
            class_id = int(box.cls[0])  # Class label
            class_name = result.names[class_id]
            
            # Create label for display
            label = f"Class {class_name}:{confidence:.2f} %"
            print(label)
            if class_name == "inlet":
                # Draw a bounding box around the "inlet" class objects
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 255, 255), 2)
                cv2.putText(frame, "Inlet", (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            else:
                # Draw a circle for other objects
                width = int(x2) - int(x1)
                height = int(y2) - int(y1)
                radius = int((width + height) / 4)
                x = int(width / 2) + int(x1)
                y = int(height / 2) + int(y1)
                cv2.circle(frame, (x, y), radius, (0, 255, 125), 2)
    
    # Display FPS on the frame
    fps_text = f"FPS: {fps:.2f}"
    cv2.putText(frame, fps_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
    
    # Display the frame with detections
    cv2.imshow("Detection Inlet", frame)
    
    # Press 'q' to exit the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close windows
cap.release()
cv2.destroyAllWindows()
