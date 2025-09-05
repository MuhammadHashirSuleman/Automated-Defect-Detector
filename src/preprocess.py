import cv2
def preprocess_image(image_path):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (640, 640))
    img = cv2.GaussianBlur(img, (5, 5), 0)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    equalized = cv2.equalizeHist(gray)
    return cv2.cvtColor(equalized, cv2.COLOR_GRAY2BGR)

def classical_defect_detection(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    equalized = cv2.equalizeHist(gray)
    edges = cv2.Canny(equalized, 50, 150)
    _, thresh = cv2.threshold(edges, 127, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    defect_img = img.copy()
    defects = []
    for cnt in contours:
        if cv2.contourArea(cnt) > 100:
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(defect_img, (x, y), (x+w, y+h), (0, 255, 0), 2)
            defects.append({"type": "unknown_defect", "area": cv2.contourArea(cnt)})
    return defect_img, defects
