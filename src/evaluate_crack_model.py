import os
import glob
from roboflow import Roboflow
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from dotenv import load_dotenv
import logging

# =============================
# SET UP LOGGING
# =============================
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# =============================
# LOAD ENV VARIABLES
# =============================
load_dotenv()
API_KEY = os.getenv("ROBOFLOW_API_KEY")  # from .env

if not API_KEY:
    logger.error("ROBOFLOW_API_KEY not found in environment variables")
    exit(1)

# =============================
# CONFIG - UPDATED WITH VALIDATION
# =============================
PROJECT_ID = "crack-detection-by-mhs-mourc"   # your Roboflow project id
VERSION_NUMBER = 1

# Use relative paths instead of absolute paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TEST_IMAGES_DIR = os.path.join(BASE_DIR, "datasets", "crack_dataset", "test", "images")
TEST_LABELS_DIR = os.path.join(BASE_DIR, "datasets", "crack_dataset", "test", "labels")

# Alternative: if you want to keep the absolute path but check if it exists
# TEST_IMAGES_DIR = r"G:\Ezitech Internship\defect-detection-app\datasets\crack_dataset\test\images"
# TEST_LABELS_DIR = r"G:\Ezitech Internship\defect-detection-app\datasets\crack_dataset\test\labels"

CONF_THRESHOLD = 0.4  # confidence threshold (float, 0–1)
# =============================

# Validate directories exist
def validate_directory(path, name):
    if not os.path.exists(path):
        logger.error(f"{name} directory does not exist: {path}")
        return False
    if not os.path.isdir(path):
        logger.error(f"{name} path is not a directory: {path}")
        return False
    return True

if not validate_directory(TEST_IMAGES_DIR, "Test Images"):
    logger.info("Creating test images directory structure...")
    os.makedirs(TEST_IMAGES_DIR, exist_ok=True)

if not validate_directory(TEST_LABELS_DIR, "Test Labels"):
    logger.info("Creating test labels directory structure...")
    os.makedirs(TEST_LABELS_DIR, exist_ok=True)

# Connect to Roboflow
logger.info("Loading Roboflow workspace...")
try:
    rf = Roboflow(api_key=API_KEY)
    logger.info("Loading Roboflow project...")
    project = rf.workspace().project(PROJECT_ID)
    model = project.version(VERSION_NUMBER).model
except Exception as e:
    logger.error(f"Failed to connect to Roboflow: {e}")
    exit(1)

# Helper to load YOLO labels (single-class: crack = 0)
def load_labels(label_path):
    labels = []
    if not os.path.exists(label_path):
        return labels
    try:
        with open(label_path, "r") as f:
            for line in f.readlines():
                parts = line.strip().split()
                if len(parts) > 0:
                    labels.append(int(parts[0]))  # class index only
    except Exception as e:
        logger.warning(f"Error reading label file {label_path}: {e}")
    return labels

y_true = []
y_pred = []

# Recursively collect all images (.jpg, .jpeg, .png)
image_extensions = ["*.jpg", "*.jpeg", "*.png"]
image_paths = []
for ext in image_extensions:
    image_paths.extend(glob.glob(os.path.join(TEST_IMAGES_DIR, "**", ext), recursive=True))

logger.info(f"Found {len(image_paths)} test images.")

if len(image_paths) == 0:
    logger.warning("No test images found. Please check TEST_IMAGES_DIR path.")
    logger.info(f"Current TEST_IMAGES_DIR: {TEST_IMAGES_DIR}")
    
    # Show what's actually in the directory
    if os.path.exists(TEST_IMAGES_DIR):
        logger.info(f"Contents of {TEST_IMAGES_DIR}:")
        for item in os.listdir(TEST_IMAGES_DIR):
            logger.info(f"  {item}")
    
    # Ask if user wants to continue with a sample image
    response = input("No images found. Do you want to download a sample image to test? (y/n): ")
    if response.lower() == 'y':
        # You could add code here to download a sample image
        logger.info("Sample image download not implemented. Please add images to the test directory.")
    exit()

# Iterate test images
for i, image_path in enumerate(image_paths):
    try:
        filename = os.path.basename(image_path)
        label_filename = os.path.splitext(filename)[0] + ".txt"
        label_path = os.path.join(TEST_LABELS_DIR, label_filename)

        # Ground truth: 1 if crack exists
        gt_labels = load_labels(label_path)
        gt = 1 if len(gt_labels) > 0 else 0

        # Prediction from Roboflow
        preds = model.predict(image_path, confidence=CONF_THRESHOLD).json()
        pred_objects = preds.get("predictions", [])
        pred = 1 if len(pred_objects) > 0 else 0

        y_true.append(gt)
        y_pred.append(pred)

        # Debug logs for first 5 images
        if i < 5:
            logger.info(f"{filename} | GT={gt} | Predictions={len(pred_objects)} | Label exists={os.path.exists(label_path)}")
    
    except Exception as e:
        logger.error(f"Error processing image {image_path}: {e}")
        continue

# =============================
# METRICS
# =============================
if len(y_true) == 0:
    logger.error("No evaluation data collected. Check dataset paths and labels.")
else:
    try:
        acc = accuracy_score(y_true, y_pred)
        prec = precision_score(y_true, y_pred, zero_division=0)
        rec = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)

        logger.info("\nModel Evaluation on Test Set")
        logger.info(f"Accuracy : {acc:.4f}")
        logger.info(f"Precision: {prec:.4f}")
        logger.info(f"Recall   : {rec:.4f}")
        logger.info(f"F1-score : {f1:.4f}")
        
        # Additional metrics
        total_images = len(y_true)
        crack_images = sum(y_true)
        detected_cracks = sum(y_pred)
        true_positives = sum([1 for i in range(len(y_true)) if y_true[i] == 1 and y_pred[i] == 1])
        
        logger.info(f"Total images: {total_images}")
        logger.info(f"Images with cracks: {crack_images}")
        logger.info(f"Images detected with cracks: {detected_cracks}")
        logger.info(f"True positives: {true_positives}")
        
    except Exception as e:
        logger.error(f"Error calculating metrics: {e}")