import cv2
import numpy as np
import math

class PoseEstimation:
    def __init__(self, frame, object_points, camera_matrix, image_points, dist_coeffs, predict_conf, model_path, cam_id):
        self.camera_frame = frame
        self.object_points = object_points
        self.camera_matrix = camera_matrix
        self.image_points = image_points
        self.dist_coeffs = dist_coeffs
        self.predict_conf = predict_conf
        self.model_path = model_path
        self.cam_id = cam_id
        self.rotation_vector = None
        self.translation_vector = None
        self.estimate_pose()

    def estimate_pose(self):
        if self.object_points is not None and self.image_points is not None:
            self.success, self.rotation_vector, self.translation_vector = cv2.solvePnP(
                self.object_points, self.image_points, self.camera_matrix, self.dist_coeffs
            )
            if self.success:
                self.display_pose()

    def display_pose(self):
        # Umwandlung des Rotationsvektors in eine Rotationsmatrix
        rotation_matrix, _ = cv2.Rodrigues(self.rotation_vector)

        # Extrahieren der Euler-Winkel aus der Rotationsmatrix
        sy = math.sqrt(rotation_matrix[0, 0] ** 2 + rotation_matrix[1, 0] ** 2)
        singular = sy < 1e-6

        if not singular:
            x_winkel = math.atan2(rotation_matrix[2, 1], rotation_matrix[2, 2])
            y_winkel = math.atan2(-rotation_matrix[2, 0], sy)
            z_winkel = math.atan2(rotation_matrix[1, 0], rotation_matrix[0, 0])
        else:
            x_winkel = math.atan2(-rotation_matrix[1, 2], rotation_matrix[1, 1])
            y_winkel = math.atan2(-rotation_matrix[2, 0], sy)
            z_winkel = 0

        # Umrechnung der Winkel in Grad
        x_winkel = round(math.degrees(x_winkel))
        self.x_deg = x_winkel
        y_winkel = round(math.degrees(y_winkel))
        self.y_deg = y_winkel
        z_winkel = round(math.degrees(z_winkel))
        self.z_deg = z_winkel
        # Anzeige der Winkel und der euklidischen Distanz
        cv2.putText(self.camera_frame, "Winkel x: " + str(x_winkel), (10, 350), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(self.camera_frame, "Winkel y: " + str(y_winkel), (10, 400), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(self.camera_frame, "Winkel z: " + str(z_winkel), (10, 450), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        euklidische_dist = np.linalg.norm(self.translation_vector)
        self.distance = euklidische_dist
        cv2.putText(self.camera_frame, "Abstand: " + str(euklidische_dist) + " mm", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Zeichnen des Koordinatensystems
        self.draw_axes()

    def draw_axes(self):
        axis_length = 50  # Länge der Achsen in mm
        axis_points = np.float32([
            [axis_length, 0, 0],  # X-Achse (rot)
            [0, axis_length, 0],  # Y-Achse (grün)
            [0, 0, axis_length],  # Z-Achse (blau)
            [0, 0, 0]             # Ursprung
        ]).reshape(-1, 3)

        # Projizieren der 3D-Achsenpunkte in das 2D-Bild
        image_points, _ = cv2.projectPoints(axis_points, self.rotation_vector, self.translation_vector, self.camera_matrix, self.dist_coeffs)


        '''
        Ursprung: Der Ursprung des Koordinatensystems befindet sich im Kameralinsen-Zentrum.
        X-Achse: Die X-Achse verläuft horizontal nach rechts.
        Y-Achse: Die Y-Achse verläuft vertikal nach unten.
        Z-Achse: Die Z-Achse verläuft senkrecht zur Bildebene und zeigt nach vorne, weg von der Kamera.
        '''
        # Zeichnen der Achsen
        origin = tuple(map(int, image_points[3].ravel()))
        x_axis = tuple(map(int, image_points[0].ravel()))
        y_axis = tuple(map(int, image_points[1].ravel()))
        z_axis = tuple(map(int, image_points[2].ravel()))

        cv2.line(self.camera_frame, origin, x_axis, (0, 0, 255), 2)  # X-Achse (rot)
        cv2.line(self.camera_frame, origin, y_axis, (0, 255, 0), 2)  # Y-Achse (grün)
        cv2.line(self.camera_frame, origin, z_axis, (255, 0, 0), 2)  # Z-Achse (blau)
