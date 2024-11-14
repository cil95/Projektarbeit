import cv2
import math
import numpy as np
# Pose Estimation Only for Inlet Webcam mode
class PoseEstimation:

    def __init__(self,camera_frame,object_points,camera_matrix,image_points,dist_coeffs,predict_conf,model_path,cam_id):
        self.camera_frame = camera_frame
        self.object_points = object_points
        self.camera_matrix = camera_matrix
        self.image_points = image_points
        self.dist_coeffs = dist_coeffs
        self.predict_conf = predict_conf
        self.model_path = model_path
        self.cam_id = cam_id
        self.calculate_pose()        
    def calculate_pose(self):
        try:
            self.success, self.rotation_vector, self.translation_vector = cv2.solvePnP(self.object_points, self.image_points, self.camera_matrix, self.dist_coeffs)
            if self.success:
                
                #print("Rotationsvektor:\n", self.rotation_vector)
                #print("Translationsvektor:\n", self.translation_vector)
                x_winkel = round(math.degrees(self.rotation_vector[0][0])) # um die x-achse
                y_winkel=round(math.degrees(self.rotation_vector[1][0])) # um die y-achse
                z_winkel =round(math.degrees(self.rotation_vector[2][0]))# um die z-achse
                cv2.putText(self.camera_frame,"Winkel x: "+str(x_winkel),(10,350),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,255,0),2)
                cv2.putText(self.camera_frame,"Winkel y: "+str(y_winkel),(10,400),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,255,0),2)
                cv2.putText(self.camera_frame,"Winkel z: "+str(z_winkel),(10,450),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,255,0),2)
                euklidische_dist = np.linalg.norm(self.translation_vector)
                cv2.putText(self.camera_frame,"Abstand: "+str(euklidische_dist),(10,50),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,255,0),2)
            else:
                print("Die PnP-Lösung war nicht erfolgreich.")
        except:
            print("Fehler bei der Poseberechnung")
            pass

