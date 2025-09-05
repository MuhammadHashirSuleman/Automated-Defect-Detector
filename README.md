# Defect Detection App

## Overview
This project implements an automated defect detection system for industrial pipelines and building structures. It detects various defects including pipeline issues (deformation, obstacles, ruptures, disconnections, misalignments, depositions) and building defects (wall cracks). The system uses YOLOv5 models for object detection combined with image preprocessing techniques to achieve high accuracy detection results.

## Features
- Web-based interface with user authentication
- Image upload and preprocessing
- Automated defect detection using YOLOv5 and YOLOV11 models
- Visualization of detected defects
- Performance metrics and evaluation reports
- Support for multiple defect types

## Project Structure
```
defect-detection-app/
├── database/              # SQLite database for user authentication
├── datasets/              # Training and testing datasets
│   ├── crack_dataset/     # Building crack detection dataset
│   └── pipeline_dataset/  # Pipeline defect detection dataset
├── models/                # Trained YOLOv5 models
│   └── pipeline_defect_detection.pt  # Pipeline defect detection model
├── results/               # Evaluation results and metrics
│   ├── crack_evaluation.txt
│   └── pipeline_evaluation.txt
├── src/                   # Source code
│   ├── app.py             # Flask web application
│   ├── detect.py          # Defect detection logic
│   ├── evaluate_crack_model.py    # Crack model evaluation
│   ├── evaluate_pipeline.py       # Pipeline model evaluation
│   ├── predict_cracks.py          # Crack detection
│   ├── preprocess.py              # Image preprocessing
│   └── static/            # Static assets for web app
├── templates/             # HTML templates for web interface
├── requirements.txt       # Project dependencies
└── README.md              # This file
```

## Model Performance

### Pipeline Defect Detection Model
```
mAP50: 0.995
Precision: 0.919 (91.9% of detections are correct)
Recall: 1.000 (100% of defects are being detected)
F1-score: 0.957 (Balanced measure of precision and recall)
```

### Crack Detection Model
```
Accuracy: 0.9966
Precision: 1.0000
Recall: 0.9966
F1-score: 0.9983
```

## Setup Instructions

### Prerequisites
- Python 3.9+
- 4GB+ RAM
- ~1GB storage

### Installation
1. Clone the repository
2. Install dependencies: `pip install -r requirements.txt`
3. Ensure the models are in the `models/` directory

## Running the Application
1. Navigate to the project directory
2. Start the web server: `python src/app.py`
3. Open a browser and go to `http://localhost:5000`
4. Login with your credentials
5. Upload an image and detect defects

## Evaluation
To evaluate the models on test datasets:

```bash
# Evaluate pipeline defect detection model
python src/evaluate_pipeline.py

# Evaluate crack detection model
python src/evaluate_crack_model.py
```

## Technologies Used
- Flask (Web Framework)
- YOLOv5 (Object Detection)
- OpenCV (Image Processing)
- PyTorch (Deep Learning)
- SQLite (Database)
- Matplotlib & Seaborn (Visualization)

## License
This project is proprietary and confidential.