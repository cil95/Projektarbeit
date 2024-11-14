from ultralytics import YOLO
import cv2
import time

class CircleDetection:
    Circle_Coordinates = {} 
    def __init__(self,model_path="runs_300/train/weights/best.pt",recognize_mode="cam",cam_id=0,image_path=None,predict_conf=0.4):
        self.predict_conf = predict_conf
        self.model_path = model_path
        if recognize_mode == "cam":
            self.cap = cv2.VideoCapture(cam_id)
            if not self.cap.isOpened():
                print(f"Error while opening camera. CAM_ID:{cam_id} not found.")
                exit()
            self.detect_per_cam()
        elif recognize_mode == "image":
            self.image = cv2.imread(image_path)
            if self.image is None:
                print("Error while opening image. Image not found. PATH:",image_path)
                exit()
            self.detect_per_image()
        
    def detect_per_cam(self):
        self.model = YOLO(self.model_path)
        # fps calc
        prev_time = 0

        while True:
            ret, frame = self.cap.read()
            if not ret:
                print("Error: Failed to capture image.")
                break
            
            curr_time = time.time()
            fps = 1 / (curr_time - prev_time)
            prev_time = curr_time

            results = self.model.predict(frame,conf=self.predict_conf)
            for result in results:
                for box in result.boxes:
                    x1, y1, x2, y2 = box.xyxy[0]
                    confidence = box.conf[0]
                    class_id = int(box.cls[0])
                    class_name = result.names[class_id]
                    label = f"Class {class_name}:{confidence:.2f} %"
                    #print(label)
                    if class_name == "inlet":
                        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 255, 255), 2)
                        cv2.putText(frame, "Inlet", (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                        self.Circle_Coordinates[class_name] = {"x": (int(x1)+ (int(x2)-int(x1))/2), "y": (int(y1)+ (int(y2)-int(y1))/2),"width":int(x2)-int(x1),"height":int(y2)-int(y1)}
                    else:
                        width = int(x2) - int(x1)
                        height = int(y2) - int(y1)
                        radius = int((width + height) / 4)
                        x = int(width / 2) + int(x1)
                        y = int(height / 2) + int(y1)
                        cv2.circle(frame, (x, y), radius, (0, 255, 125), 2)
                        self.Circle_Coordinates[class_name] = {"x": x, "y": y,"radius":radius}
            # Display FPS on the frame
            fps_text = f"FPS: {fps:.2f}"
            cv2.putText(frame, fps_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
            
            cv2.imshow("Detection Circles Webcam",frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        self.cap.release()
        cv2.destroyAllWindows()

    def detect_per_image(self):
        self.model = YOLO(self.model_path)
        results = self.model.predict(self.image)        # without confidence manifestation
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = box.xyxy[0]
                confidence = box.conf[0]
                class_id = int(box.cls[0])
                class_name = result.names[class_id]
                label = f"Class {class_name}:{confidence:.2f} %"
                #print(label)
                if class_name == "inlet":
                    cv2.rectangle(self.image, (int(x1), int(y1)), (int(x2), int(y2)), (255, 255, 255), 2)
                    cv2.putText(self.image, "Inlet", (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    self.Circle_Coordinates[class_name] = {"x": (int(x1)+ (int(x2)-int(x1))/2), "y": (int(y1)+ (int(y2)-int(y1))/2),"width":int(x2)-int(x1),"height":int(y2)-int(y1)}
                else:
                    width = int(x2) - int(x1)
                    height = int(y2) - int(y1)
                    radius = int((width + height) / 4)
                    x = int(width / 2) + int(x1)
                    y = int(height / 2) + int(y1)
                    cv2.circle(self.image, (x, y), radius, (0, 255, 125), 2)
                    self.Circle_Coordinates[class_name] = {"x": x, "y": y,"radius":radius}
        cv2.imshow("Detection Circles Image",self.image)
        cv2.waitKey(0)

#webcam = CircleDetection(recognize_mode="cam",cam_id=0)
# image = CircleDetection(recognize_mode="image",image_path="./assets/inlet.jpeg")
# print(image.Circle_Coordinates)
