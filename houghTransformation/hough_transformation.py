import cv2
import numpy as np

# Hough Circle Transformation anwenden
def nothing(x):
    pass

# Create trackbars for parameter adjustment
cv2.namedWindow('Parameters')
cv2.createTrackbar('dp', 'Parameters', 12, 20, nothing)
cv2.createTrackbar('minDist', 'Parameters', 30, 100, nothing)
cv2.createTrackbar('param1', 'Parameters', 50, 100, nothing)
cv2.createTrackbar('param2', 'Parameters', 30, 100, nothing)
cv2.createTrackbar('minRadius', 'Parameters', 100, 200, nothing)
cv2.createTrackbar('maxRadius', 'Parameters', 120, 200, nothing)

# Webcam initialisieren
cap = cv2.VideoCapture(0)
while True:
    # Frame von der Webcam lesen
    ret, frame = cap.read()
    if not ret:
        break
    # convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # reduce noise
    gray = cv2.medianBlur(gray, 5)


    # Get current positions of trackbars
    dp = cv2.getTrackbarPos('dp', 'Parameters') / 10
    minDist = cv2.getTrackbarPos('minDist', 'Parameters')
    param1 = cv2.getTrackbarPos('param1', 'Parameters')
    param2 = cv2.getTrackbarPos('param2', 'Parameters')
    minRadius = cv2.getTrackbarPos('minRadius', 'Parameters')
    maxRadius = cv2.getTrackbarPos('maxRadius', 'Parameters')
    circles = cv2.HoughCircles(gray, # grayscale image
                            cv2.HOUGH_GRADIENT, # detection method
                            dp=dp, # inverse ratio of the accumulator resolution to the image resolution
                            minDist=minDist, # minimum distance between the centers of the detected circles
                            param1=param1, # higher threshold for the Canny edge detector
                            param2=param2, # accumulator threshold for the circle centers at the detection stage
                            minRadius=minRadius, # minimum circle radius
                            maxRadius=maxRadius) # maximum circle radius
    

    # Kreise zeichnen
    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")
        for (x, y, r) in circles:
            cv2.circle(frame, (x, y), r, (0, 255, 0), 4)
            cv2.rectangle(frame, (x - 5, y - 5), (x + 5, y + 5), (0, 128, 255), -1)

    # Frame anzeigen
    cv2.imshow("Detected Circles", frame)

    # Beenden, wenn 'q' gedrückt wird
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Ressourcen freigeben
cap.release()
cv2.destroyAllWindows()