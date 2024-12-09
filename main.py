import math
from ultralytics import YOLO
import cv2
import time
import numpy as np
from pose_estimation import PoseEstimation
class CircleDetection:
    Circle_Coordinates = {} 
    def __init__(self,model_path="runs_300/train/weights/best.pt",recognize_mode="cam",cam_id=0,image_path=None,predict_conf=0.4, object_points=None,camera_matrix=None,dist_coeffs=None):
        self.object_points = object_points
        self.camera_matrix = camera_matrix
        self.dist_coeffs = dist_coeffs
        self.predict_conf = predict_conf
        self.model_path = model_path
        self.cam_id = cam_id
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
            self.Circle_Coordinates = {}


            results = self.model.predict(frame,conf=self.predict_conf)
            circle_list = []
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
                    elif class_name == "inlet-plus" or class_name == "inlet-minus":
                        # ignore it
                        continue
                    else:
                        width = int(x2) - int(x1)
                        height = int(y2) - int(y1)
                        radius = int((width + height) / 4)
                        x = int(width / 2) + int(x1)
                        y = int(height / 2) + int(y1)
                        circle_list.append(((x,y),radius))

            # Display FPS on the frame
            fps_text = f"FPS: {fps:.2f}"
            cv2.putText(frame, fps_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
            errorLabel = "Error: Failed to detect the circles in the image."
            try: 
              self.get_circle_coordinates(frame,circle_list,self.Circle_Coordinates["inlet"])
            except:
                cv2.putText(frame, errorLabel, (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 0, 0), 1)


            # verify the detection of the circles            
            classes = ["inlet","inlet-center","inlet-center-left","inlet-center-right","inlet-above-left","inlet-above-right","inlet-below-left","inlet-below-right"]
            # Do the Pose Estimation
            if all(elem in self.Circle_Coordinates.keys() for elem in classes):
                # set the imagepoints
                self.image_points = np.array([
                    [self.Circle_Coordinates["inlet-center"]["x"], self.Circle_Coordinates["inlet-center"]["y"]],          # PE (Mittelpunkt)
                    [self.Circle_Coordinates["inlet-center-left"]["x"], self.Circle_Coordinates["inlet-center-left"]["y"]],          # Linker mittlerer Kreis
                    [self.Circle_Coordinates["inlet-center-right"]["x"], self.Circle_Coordinates["inlet-center-right"]["y"]],         # Rechter mittlerer Kreis
                    [self.Circle_Coordinates["inlet-above-left"]["x"], self.Circle_Coordinates["inlet-above-left"]["y"]],          # Oberer linker Kreis
                    [self.Circle_Coordinates["inlet-above-right"]["x"], self.Circle_Coordinates["inlet-above-right"]["y"]],         # Oberer rechter Kreis
                    [self.Circle_Coordinates["inlet-below-left"]["x"], self.Circle_Coordinates["inlet-below-left"]["y"]],         # Unterer linker Kreis
                    [self.Circle_Coordinates["inlet-below-right"]["x"], self.Circle_Coordinates["inlet-below-right"]["y"]]         # Unterer rechter Kreis
                ], dtype=np.float32)

                pose = PoseEstimation(frame,self.object_points,self.camera_matrix,self.image_points,self.dist_coeffs,self.predict_conf,self.model_path,self.cam_id)
                print(pose.rotation_vector)
                print(pose.translation_vector)
            else:
                cv2.putText(frame, "Pose Estimation Failed - key Error ", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 0), 1)
            cv2.imshow("Detection Circles Webcam",frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        self.cap.release()
        cv2.destroyAllWindows()

    def calculate_angle(self,point1,point2):
        x1,y1 = point1
        x2,y2 = point2
        angle = math.atan2(y2-y1,x2-x1)
        return round(math.degrees(angle),2)

    def find_circle_center(self,circle_list):
        # first get the arithmetic values
        length = len(circle_list)
        if length == 0:
            return None
        
        x=0
        y=0
        radius = 0
        for circle in circle_list:
            x += circle[0][0]
            y += circle[0][1]
            radius += circle[1]
        x_arith = x/length
        y_arith = y/length
        radius_arith = radius/length

        # find the point in circle_list which is nearest to the arithmetic values
        nearest_circle = None
        min_distance = float('inf')
        for circle in circle_list:
            (x, y), radius = circle
            # Berechne die euklidische Distanz zwischen den Mittelpunkten
            distance = ((x - x_arith) ** 2 + (y - y_arith) ** 2) ** 0.5
            # Optional: Berücksichtige auch den Unterschied im Radius
            radius_diff = abs(radius - radius_arith)
            total_distance = distance + radius_diff

            if total_distance < min_distance:
                min_distance = total_distance
                nearest_circle = circle

        return nearest_circle
    
    def find_nearest_y(self,circle_list,circle_center):
        # find the nearest circle to the center in y direction
        min_distance = float('inf')
        nearest_circle = None
        for circle in circle_list:
            (x, y), radius = circle
            distance = abs(y - circle_center[0][1])
            if distance < min_distance:
                min_distance = distance
                nearest_circle = circle
        return nearest_circle
    
    def get_circle_coordinates(self,image,circle_list,inlet):
            # kamera kos
            img_height,img_width = image.shape[:2]
            center_image_x = img_width//2
            center_image_y = img_height//2
            # cv2.line(image, (int(center_image_x), 0), (int(center_image_x), img_height), (0, 0, 255), 2)
            # cv2.line(image, (0, int(center_image_y)), (img_width, int(center_image_y)), (0, 0, 255), 2)

            inlet_x, inlet_y = inlet["x"], inlet["y"]
            inlet_width, inlet_height = inlet["width"], inlet["height"]
            
            # # # insert an arrow for location of inlet
            # # if inlet_x < center_image_x and abs(inlet_x - center_image_x) >= 15:
            # #     cv2.arrowedLine(image, (center_image_x + 200,center_image_y), (center_image_x+100,center_image_y), (255, 0, 0), 10)
            # # elif inlet_x > center_image_x and abs(inlet_x - center_image_x) >= 15:
            # #     cv2.arrowedLine(image, (center_image_x - 200,center_image_y), (center_image_x-100,center_image_y), (255, 0, 0), 10)
            
            # # if inlet_y < center_image_y and abs(inlet_y - center_image_y) >= 15:
            # #     cv2.arrowedLine(image, (center_image_x,center_image_y + 200), (center_image_x,center_image_y+100), (255, 0, 0), 10)
            # # elif inlet_y > center_image_y and abs(inlet_y - center_image_y) >= 15:
            # #     cv2.arrowedLine(image, (center_image_x,center_image_y - 200), (center_image_x,center_image_y-100), (255, 0, 0), 10)
                

            filtered_circle_list = []
            for circle in circle_list:
                (x, y), radius = circle
                if (inlet_x - ((inlet_width+50) / 2)) < x < (inlet_x + ((inlet_width+50) / 2)) and (inlet_y - ((inlet_height+50) / 2)) < y < (inlet_y + ((inlet_height+50) / 2)):
                    filtered_circle_list.append(circle)

            circle_list = filtered_circle_list
            if circle_list is not None :
                # find the center of the circles
                center = self.find_circle_center(circle_list)
                self.Circle_Coordinates["inlet-center"] = {"x": center[0][0], "y": center[0][1],"radius":center[1]}
                # first filter the mid circles
                circle_list.remove(center)
                for _ in range(2):
                    if "inlet-circle-right" in self.Circle_Coordinates.keys() and "inlet-circle-left" in self.Circle_Coordinates.keys():
                        break
                    circle_mid = self.find_nearest_y(circle_list,center)
                    if circle_mid[0][0]> center[0][0]:
                        self.Circle_Coordinates["inlet-center-right"] = {"x": circle_mid[0][0], "y": circle_mid[0][1],"radius":circle_mid[1]}
                        circle_list.remove(circle_mid)
                    elif circle_mid[0][0]< center[0][0]:
                        self.Circle_Coordinates["inlet-center-left"] = {"x": circle_mid[0][0], "y": circle_mid[0][1],"radius":circle_mid[1]}
                        circle_list.remove(circle_mid)
                
                for circle in circle_list:
                    # above circles
                    if circle[0][1] < center[0][1]:
                        if circle[0][0] < center[0][0]:
                            self.Circle_Coordinates["inlet-above-left"] = {"x": circle[0][0], "y": circle[0][1],"radius":circle[1]}
                        else:
                            self.Circle_Coordinates["inlet-above-right"] = {"x": circle[0][0], "y": circle[0][1],"radius":circle[1]}
                    # below circles
                    if circle[0][1] > center[0][1]:
                        if circle[0][0] < center[0][0]:
                            self.Circle_Coordinates["inlet-below-left"] = {"x": circle[0][0], "y": circle[0][1],"radius":circle[1]}
                        else:
                            self.Circle_Coordinates["inlet-below-right"] = {"x": circle[0][0], "y": circle[0][1],"radius":circle[1]}
            
            for circle_class in self.Circle_Coordinates:
                inlet_center_point = (self.Circle_Coordinates["inlet-center"]["x"],self.Circle_Coordinates["inlet-center"]["y"])
                if circle_class != "inlet":
                    cv2.putText(image, circle_class, (self.Circle_Coordinates[circle_class]["x"], self.Circle_Coordinates[circle_class]["y"] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                    cv2.circle(image, (self.Circle_Coordinates[circle_class]["x"], self.Circle_Coordinates[circle_class]["y"]), self.Circle_Coordinates[circle_class]["radius"], (0, 255, 125), 2)
                # validate Coordinates
                elif circle_class == "inlet-center-right":
                    inlet_center_right = (self.Circle_Coordinates["inlet-center-right"]["x"],self.Circle_Coordinates["inlet-center-right"]["y"])
                    angle = self.calculate_angle(inlet_center_point,inlet_center_right)
                    self.Circle_Coordinates[circle_class]["angle"] = angle
                    if angle > 175 and angle < 185:
                        cv2.circle(image, (self.Circle_Coordinates[circle_class]["x"], self.Circle_Coordinates[circle_class]["y"]), self.Circle_Coordinates[circle_class]["radius"], (0, 255, 125), 2)
                elif circle_class == "inlet-center-left":
                    inlet_center_left = (self.Circle_Coordinates["inlet-center-left"]["x"],self.Circle_Coordinates["inlet-center-left"]["y"])
                    angle = self.calculate_angle(inlet_center_point,inlet_center_left)
                    self.Circle_Coordinates[circle_class]["angle"] = angle
                    if angle >-5 and angle < 5:
                        cv2.circle(image, (self.Circle_Coordinates[circle_class]["x"], self.Circle_Coordinates[circle_class]["y"]), self.Circle_Coordinates[circle_class]["radius"], (0, 255, 125), 2)
                elif circle_class == "inlet-above-left":
                    inlet_above_left = (self.Circle_Coordinates["inlet-above-left"]["x"],self.Circle_Coordinates["inlet-above-left"]["y"])
                    angle = self.calculate_angle(inlet_center_point,inlet_above_left)
                    self.Circle_Coordinates[circle_class]["angle"] = angle
                    if angle > 50 and angle < 57:
                        cv2.circle(image, (self.Circle_Coordinates[circle_class]["x"], self.Circle_Coordinates[circle_class]["y"]), self.Circle_Coordinates[circle_class]["radius"], (0, 255, 125), 2)
                elif circle_class == "inlet-above-right":
                    inlet_above_right = (self.Circle_Coordinates["inlet-above-right"]["x"],self.Circle_Coordinates["inlet-above-right"]["y"])
                    angle = self.calculate_angle(inlet_center_point,inlet_above_right)
                    self.Circle_Coordinates[circle_class]["angle"] = angle
                    if angle > 120 and angle < 130:
                        cv2.circle(image, (self.Circle_Coordinates[circle_class]["x"], self.Circle_Coordinates[circle_class]["y"]), self.Circle_Coordinates[circle_class]["radius"], (0, 255, 125), 2)
                elif circle_class == "inlet-below-left":
                    inlet_below_left = (self.Circle_Coordinates["inlet-below-left"]["x"],self.Circle_Coordinates["inlet-below-left"]["y"])
                    angle = self.calculate_angle(inlet_center_point,inlet_below_left)
                    self.Circle_Coordinates[circle_class]["angle"] = angle
                    if angle > -55 and angle < -65:
                        cv2.circle(image, (self.Circle_Coordinates[circle_class]["x"], self.Circle_Coordinates[circle_class]["y"]), self.Circle_Coordinates[circle_class]["radius"], (0, 255, 125), 2)
                elif circle_class == "inlet-below-right":
                    inlet_below_right = (self.Circle_Coordinates["inlet-below-right"]["x"],self.Circle_Coordinates["inlet-below-right"]["y"])
                    angle = self.calculate_angle(inlet_center_point,inlet_below_right)
                    self.Circle_Coordinates[circle_class]["angle"] = angle
                    if angle > -115 and angle < -125:
                        cv2.circle(image, (self.Circle_Coordinates[circle_class]["x"], self.Circle_Coordinates[circle_class]["y"]), self.Circle_Coordinates[circle_class]["radius"], (0, 255, 125), 2)

    def detect_per_image(self):
        self.model = YOLO(self.model_path)
        results = self.model.predict(self.image)        # without confidence manifestation
        circle_list = []
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = box.xyxy[0]
                confidence = box.conf[0]
                class_id = int(box.cls[0])
                class_name = result.names[class_id]
                label = f"Class {class_name}:{confidence:.2f} %"
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
                    circle_list.append(((x,y),radius))
        try:
            self.get_circle_coordinates(self.image,circle_list,self.Circle_Coordinates["inlet"])
        except Exception as e:
            print("Failed to detect the circles in the image. ERROR:",e)
        cv2.imshow("Detection Circles Image",self.image)
        cv2.waitKey(0)



# # Object Points in 3D | Maße aus Normblatt
# object_points = np.array([
#         [0, 0, 0],          # PE (Mittelpunkt)
#         [-16, 0, 0],        # Linker mittlerer Kreis
#         [16, 0, 0],         # Rechter mittlerer Kreis
#         [-8, 11.2, 0],      # Oberer linker Kreis
#         [8, 11.2, 0],       # Oberer rechter Kreis
#         [-8, -13.9, 0],     # Unterer linker Kreis
#         [8, -13.9, 0]       # Unterer rechter Kreis
#         ], dtype=np.float32)

# Maße vom gebastelten inlet
object_points = np.array([
        [0, 0, 0],          # PE (Mittelpunkt)
        [-26, 0, 0],        # Linker mittlerer Kreis
        [26, 0, 0],         # Rechter mittlerer Kreis
        [-8, 18 ,0],      # Oberer linker Kreis
        [8, 18, 0],       # Oberer rechter Kreis
        [-12, -22, 0],     # Unterer linker Kreis
        [12, -22, 0]       # Unterer rechter Kreis
        ], dtype=np.float32)

# # camera matrix webcam 
camera_matrix = np.array([
    [2.10965926e+03, 0, 9.48821286e+02],
    [0, 1.99932864e+03, 5.93297103e+02],
    [0, 0, 1.00000000e+00]
], dtype=np.float32)

# # camera matrix Iphone 14 plus 
# camera_matrix = np.array([
#     [1.23769211e+03, 0, 4.35258529e+02],
#     [0, 1.30550723e+03, 9.02925725e+02],
#     [0, 0, 1.00000000e+00]
# ], dtype=np.float32)


# Keine Verzerrung der Kamera
dist_coeffs = np.zeros(4)  # Keine Verzerrung


CircleDetection(model_path="runs_300_rotation2/train/weights/best.pt",recognize_mode="cam",cam_id=1,predict_conf=0.4,
                object_points=object_points,camera_matrix=camera_matrix,dist_coeffs=dist_coeffs) #  model with 300 epocs
