"Yolo ohne api von roboflow aber nur bilderkknunng keine webcam"
from ultralytics import YOLO
import cv2
image = cv2.imread("./assets/inlet4.jpg")
# Load the YOLO model
#model = YOLO("./runs/detect/train/weights/best.pt") with 200 epochs
model = YOLO("runs_300/train/weights/best.pt") # with 300 epochs

# Run predictions on the image
results = model.predict("./assets/inlet4.jpg")

# Extract and print the bounding box data
for result in results:
    for box in result.boxes:
        # Get the x1, y1 (top-left) and x2, y2 (bottom-right) coordinates
        x1, y1, x2, y2 = box.xyxy[0]  # Get coordinates of the first detected object
        # die texte
        confidence = box.conf[0]  # Confidence score
        class_id = int(box.cls[0])  # Class label
        class_name=result.names[class_id]
        label = f"Class {class_name}: {confidence:.2f} %"
        print(label)
        if class_name == "inlet":
            cv2.rectangle(image,(int(x1),int(y1)),(int(x2),int(y2)),(255,255,255),2)
            cv2.putText(image, "Inlet", (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        else:
            # Calculate width and height
            width = int(x2) - int(x1)
            height = int(y2) - int(y1)
            radius = int((width+height)/4)
            x = int(width/2)+int(x1)
            y=int(height/2)+int(y1)
            cv2.circle(image,(x,y),radius,(0,255,125),1)


cv2.imshow("Detection",image)
cv2.waitKey(0)

