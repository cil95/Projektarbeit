import cv2
import numpy as np
import scipy.spatial.distance as distance
def contrast_brightness(contrastlevel,brightnesslevel,imageToEdit):
    return cv2.addWeighted(imageToEdit,contrastlevel,np.zeros(imageToEdit.shape,imageToEdit.dtype),0,brightnesslevel)



cap = cv2.VideoCapture(0)
desired_width = 540
desired_height = 380
cap.set(cv2.CAP_PROP_FRAME_WIDTH, desired_width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, desired_height)
def nothing(x):
    pass

cv2.namedWindow('Parameters')
cv2.createTrackbar('Threshold', 'Parameters', 1, 255, nothing)
cv2.createTrackbar('Radius', 'Parameters', 10, 1000, nothing)
cv2.createTrackbar("Contrast", 'Parameters', 1, 100, nothing)
cv2.createTrackbar("Brightness", 'Parameters', 1, 100, nothing)


def find_nearest_circles(circles,num = 8):
    if num <= len(circles):
        centers = np.array([circle[0] for circle in circles])
        dist_matrix = distance.cdist(centers, centers, 'euclidean')
        np.fill_diagonal(dist_matrix, np.inf)
        nearest_indices = np.unravel_index(np.argsort(dist_matrix, axis=None)[:num], dist_matrix.shape)
        nearest_circles = [(circles[i], circles[j]) for i, j in zip(*nearest_indices)]
        return nearest_circles
    else:
        return None


while True:
    ret, frame = cap.read()
    if not ret:
        break

    perfect_thresh=cv2.getTrackbarPos('Threshold', 'Parameters')
    contrastlevel=cv2.getTrackbarPos('Contrast', 'Parameters')
    brightnesslevel=cv2.getTrackbarPos('Brightness', 'Parameters')
    frame=contrast_brightness(contrastlevel,brightnesslevel,frame)
    # threshold
    _, thresh = cv2.threshold(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                                                ,perfect_thresh,
                                                255,
                                                cv2.THRESH_BINARY)

    # find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    circles = []
    for contour in contours:
        # Finde den minimal umschreibenden Kreis
        (x, y), radius = cv2.minEnclosingCircle(contour)
        center = (int(x), int(y))
        radius = int(radius)

        # Berechne den Konturbereich und Kreisbereich
        contour_area = cv2.contourArea(contour)
        circle_area = np.pi * (radius ** 2)

        if circle_area > 1.0 and radius > cv2.getTrackbarPos('Radius', 'Parameters'):
            circles.append((center, radius))
    try:
        nearest_circles = find_nearest_circles(circles)
        for circle in nearest_circles:
            cv2.circle(frame, circle[0][0], circle[0][1], (0, 0, 255), 2)
    except:
        pass
    cv2.imshow("Contur kreise",frame)
    cv2.imshow("Binary Frame",thresh)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break