import cv2
import numpy as np

# List of template images
template_paths = [
    "./assets/mitte/mitte_wiki.jpeg",
    "./assets/mitte/template_gebastelt.jpeg",
]

# Load templates
templates = []
for path in template_paths:
    template = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    assert template is not None, f"Template image not found: {path}"
    templates.append((template, template.shape[::-1],path))

cap = cv2.VideoCapture(0)
# Set the desired width and height
desired_width = 540
desired_height = 380
cap.set(cv2.CAP_PROP_FRAME_WIDTH, desired_width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, desired_height)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    methods = ['cv2.TM_CCOEFF', 'cv2.TM_CCOEFF_NORMED']

    for template, (w_temp, h_temp), path in templates:
        for method in methods:
            result = cv2.matchTemplate(frame_gray, template, eval(method))
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
            if method in ['cv2.TM_SQDIFF', 'cv2.TM_SQDIFF_NORMED']:
                top_left = min_loc
            else:
                top_left = max_loc
            bottom_right = (top_left[0] + w_temp, top_left[1] + h_temp)
            cv2.rectangle(frame, top_left, bottom_right, (0, 255, 0), 2)
            cv2.putText(frame, path.split('/')[-1], (top_left[0], top_left[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            

            # Sicherstellen, dass der Text innerhalb des Bildes bleibt
            cv2.imshow(method, frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
