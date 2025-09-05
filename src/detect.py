import cv2
import numpy as np
import onnxruntime as ort
from ultralytics import YOLO
from preprocess import classical_defect_detection

def detect_defects(image_path, preprocessed_img, detection_type="pipeline"):
    img = cv2.imread(image_path)
    img_resized = cv2.resize(img, (640, 640))
    
    if detection_type == "pipeline":
        model_path = "models/pipeline_defect_detection.pt"
        model = YOLO(model_path)
        results = model(img_resized, verbose=False)[0]
        xyxy = results.boxes.xyxy.cpu().numpy()
        conf = results.boxes.conf.cpu().numpy()
        cls = results.boxes.cls.cpu().numpy()
        boxes = xyxy[None, :]
        scores = conf[None, :]
        labels = cls[None, :]
    elif detection_type == "crack":
        model_path = "models/crack_rtdetr.onnx"
        session = ort.InferenceSession(model_path)
        input_name = session.get_inputs()[0].name
        img_input = img_resized.transpose(2, 0, 1).astype(np.float32) / 255.0
        img_input = np.expand_dims(img_input, axis=0)
        outputs = session.run(None, {input_name: img_input})
        boxes, scores, labels = outputs[0], outputs[1], outputs[2]
    else:
        raise ValueError("Invalid detection type")
    
    detected_img = img_resized.copy()
    defects = []
    coco_names = ['Deformation', 'Obstacle', 'Rupture', 'Disconnect', 'Misalignment', 'Deposition']  # Classes from trained model
    for box, score, label in zip(boxes[0], scores[0], labels[0]):
        if score > 0.5:
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(detected_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            defects.append({"type": coco_names[int(label)] if int(label) < len(coco_names) else "unknown", "conf": float(score)})
    
    return detected_img, defects
