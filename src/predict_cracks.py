import cv2
from roboflow import Roboflow
import os
from dotenv import load_dotenv

load_dotenv()

# Load Roboflow model
def load_model():
    api_key = os.getenv('ROBOFLOW_API_KEY')
    if not api_key:
        raise ValueError("ROBOFLOW_API_KEY not set in environment")
    rf = Roboflow(api_key=api_key)
    project = rf.workspace().project("crack-detection-by-mhs-mourc")
    model = project.version(1).model
    return model

model = None

def detect_cracks(image_path):
    global model
    if model is None:
        model = load_model()
    prediction = model.predict(image_path, confidence=40, overlap=30).json()
    img = cv2.imread(image_path)
    defects = []
    for pred in prediction['predictions']:
        x, y, w, h = pred['x'], pred['y'], pred['width'], pred['height']
        conf = pred['confidence']
        class_type = pred['class']
        cv2.rectangle(img, (int(x - w/2), int(y - h/2)), (int(x + w/2), int(y + h/2)), (0, 0, 255), 2)
        cv2.putText(img, f"{class_type}: {conf:.2f}", (int(x - w/2), int(y - h/2) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
        defects.append({'type': class_type, 'conf': conf})
    
    return img, defects
