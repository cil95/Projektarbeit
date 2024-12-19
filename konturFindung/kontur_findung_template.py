import cv2
import numpy as np
import scipy.spatial.distance as distance
def contrast_brightness(contrastlevel,brightnesslevel,imageToEdit):
    return cv2.addWeighted(imageToEdit,contrastlevel,np.zeros(imageToEdit.shape,imageToEdit.dtype),0,brightnesslevel)


# template

template = cv2.imread("./assets/mitte_wiki.jpeg", cv2.IMREAD_GRAYSCALE) 


cap = cv2.VideoCapture(0)
desired_width = 540
desired_height = 380
cap.set(cv2.CAP_PROP_FRAME_WIDTH, desired_width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, desired_height)
def nothing(x):
    pass

cv2.namedWindow('Parameters')
cv2.createTrackbar('Threshold', 'Parameters', 1, 255, nothing)
cv2.createTrackbar('Min Radius', 'Parameters', 10, 1000, nothing)
cv2.createTrackbar('Max Radius', 'Parameters', 10, 1000, nothing)
cv2.createTrackbar("Contrast", 'Parameters', 1, 100, nothing)
cv2.createTrackbar("Brightness", 'Parameters', 1, 100, nothing)

# eine funktion um den winkel zwischen zwei kreismittelpunkten zu bestimmen
def angle_between_points(center1, center2):
    x1, y1 = center1
    x2, y2 = center2
    return np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi


def sort_circles(circles,center_template):
    circles = [circle for circle in circles if distance.euclidean(circle[0], center_template) > 7.0]
    # sortiere die Kreise nach der Entfernung zum Mittelpunkt
    circles = sorted(circles, key=lambda circle: distance.euclidean(circle[0], center_template))
    circles_angle_sorted = []
    # sortiere nach winkel
    angle = (angle_between_points(center_template, circle[0]))
    for circle in circles:
        print(angle)
        if -5< angle < 5 or 178 < angle < 182:
            circles_angle_sorted.append(circle)
        elif 55 < angle < 120:
            circles_angle_sorted.append(circle)
        elif -130 < angle < -55:
            circles_angle_sorted.append(circle)
    return circles_angle_sorted



def find_nearest_circles(circles, center_template, num=8):
    if len(circles) < num:
        return circles
    centers = np.array([circle[0] for circle in circles])
    dists = distance.cdist([center_template], centers, 'euclidean')[0]
    nearest_indices = np.argsort(dists)[:num]
    nearest_circles = [circles[i] for i in nearest_indices]
    nearest_circles = sort_circles(nearest_circles, center_template)
    return nearest_circles


def template_matching(gray_img, gray_template):
    result = cv2.matchTemplate(gray_img, gray_template, cv2.TM_CCOEFF_NORMED)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result) # finde die entsprechenden werte

    # Berechne die Breite und Höhe des gefundenen Bereichs
    width = gray_template.shape[1]
    height = gray_template.shape[0]
    radius = (width + height) // 4

    top_left = max_loc  # oben links ist der am übereinstimmste punkt
    bottom_right = (top_left[0] + template.shape[1], top_left[1] + template.shape[0]) # wie kommt man dadrauf? -> template.shape [height,width,layer], max_loc[x,y] top_left[0] = max_loc xwert
    # cv2.rectangle(img, top_left, bottom_right, (0, 0, 255), 2)
    # mittelpunkt des rechtecks
    center_x = top_left[0] + template.shape[1] // 2
    center_y = top_left[1] + template.shape[0] // 2

    return (center_x, center_y), radius


def video_processing(cap):
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        center_template,radius_template=template_matching(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), template)
        cv2.circle(frame, center_template, radius_template, (0, 0, 255), 2)
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

            # Kreisfläche
            circle_area = np.pi * (radius ** 2)

            if circle_area > 1.0 and radius > cv2.getTrackbarPos('Min Radius', 'Parameters') and radius < cv2.getTrackbarPos('Max Radius', 'Parameters'):
                circles.append((center, radius))
        try:
            nearest_circles = find_nearest_circles(circles,center_template)
            for circle in nearest_circles:
                cv2.circle(frame, circle[0], circle[1], (0, 255, 0), 2)
        except Exception as e:
            print(e)
            
        cv2.imshow("Konturfindung + Template Matching",frame)
        cv2.imshow("Binary Frame",thresh)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

def main():
    video_processing(cap)

if __name__ == "__main__":
    main()
