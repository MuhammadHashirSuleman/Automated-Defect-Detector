import os
import sys
import torch
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import cv2
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

class ModelEvaluator:
    def __init__(self, model_path, data_path):
        """
        Initialize model evaluator
        
        Args:
            model_path: Path to trained .pt model
            data_path: Path to dataset config.yaml
        """
        self.model_path = model_path
        self.data_path = data_path
        self.model = None
        self.class_names = {
            0: "Deformation",
            1: "Obstacle",
            2: "Rupture",
            3: "Disconnect",
            4: "Misalignment",
            5: "Deposition"
        }
        self.results = {}
        
    def load_model(self):
        """Load the trained model"""
        logging.info("🔄 Loading model...")
        try:
            # Load YOLOv5 model from torch hub with force_reload=True to avoid cache issues
            self.model = torch.hub.load('ultralytics/yolov5', 'custom', path=self.model_path, force_reload=True)
            self.model.conf = 0.5  # Confidence threshold
            self.model.iou = 0.6   # IoU threshold
            logging.info(f"✅ Model loaded successfully")
            logging.info(f"Classes: {self.class_names}")
            return self.model
        except Exception as e:
            logging.error(f"Error loading model: {e}")
            sys.exit(1)
    
    def evaluate_model(self, test_dir=None):
        """Evaluate model on test images"""
        logging.info("\n📊 Evaluating model performance...")
        
        if test_dir is None:
            # Use validation directory from config
            test_dir = os.path.join(os.path.dirname(self.data_path), "images/images/val")
        
        if not os.path.exists(test_dir):
            logging.error(f"Test directory not found: {test_dir}")
            return None
            
        logging.info(f"Using test images from: {test_dir}")
        
        # Get all test images
        test_images = [os.path.join(test_dir, f) for f in os.listdir(test_dir) 
                      if f.endswith(('.jpg', '.jpeg', '.png'))]
        
        if not test_images:
            logging.error("No test images found")
            return None
            
        logging.info(f"Found {len(test_images)} test images")
        
        # Process each image and collect results
        all_predictions = []
        all_ground_truth = []
        
        for img_path in test_images:
            # Get ground truth labels
            gt_labels = self._get_ground_truth_labels(img_path)
            
            # Get predictions
            results = self.model(img_path)
            pred_classes = [int(pred[5]) for pred in results.xyxy[0] if float(pred[4]) >= 0.5]
            
            # Store binary classification (defect present or not)
            has_defect_gt = len(gt_labels) > 0
            has_defect_pred = len(pred_classes) > 0
            
            all_ground_truth.append(1 if has_defect_gt else 0)
            all_predictions.append(1 if has_defect_pred else 0)
            
        # Calculate metrics
        self.results = self._calculate_metrics(all_predictions, all_ground_truth)
        return self.results
    
    def _get_ground_truth_labels(self, image_path):
        """Get ground truth labels for an image"""
        # Convert image path to label path
        # Example: images/images/val/001.jpg -> labels/labels/val/001.txt
        label_path = image_path.replace('images/images', 'labels/labels').replace(
            os.path.splitext(image_path)[1], '.txt')
        
        if not os.path.exists(label_path):
            return []
            
        # Read labels
        labels = []
        with open(label_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 1:
                    class_id = int(parts[0])
                    labels.append(class_id)
        return labels
    
    def _calculate_metrics(self, predictions, ground_truth):
        """Calculate evaluation metrics"""
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
        
        # Calculate metrics
        metrics = {
            'accuracy': accuracy_score(ground_truth, predictions),
            'precision': precision_score(ground_truth, predictions, zero_division=0),
            'recall': recall_score(ground_truth, predictions, zero_division=0),
            'f1_score': f1_score(ground_truth, predictions, zero_division=0),
            'confusion_matrix': confusion_matrix(ground_truth, predictions)
        }
        
        return metrics
        
    def print_detailed_metrics(self):
        """Print detailed evaluation metrics"""
        if not self.results:
            logging.error("❌ No metrics available. Run evaluate_model() first.")
            return
            
        logging.info("\n" + "="*60)
        logging.info("📈 DETAILED EVALUATION METRICS")
        logging.info("="*60)
        
        # Overall metrics
        logging.info(f"\n🎯 Overall Performance:")
        logging.info(f"   Accuracy: {self.results['accuracy']:.4f}")
        logging.info(f"   Precision: {self.results['precision']:.4f}")
        logging.info(f"   Recall: {self.results['recall']:.4f}")
        logging.info(f"   F1-Score: {self.results['f1_score']:.4f}")
        
        # Save results to file
        results_path = os.path.join(os.path.dirname(os.path.dirname(self.model_path)), 
                                   "results", "pipeline_evaluation.txt")
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(results_path), exist_ok=True)
        
        with open(results_path, 'w') as f:
            f.write("PIPELINE MODEL EVALUATION RESULTS\n")
            f.write("=================================\n\n")
            f.write(f"Model: {self.model_path}\n")
            f.write(f"Dataset: {self.data_path}\n\n")
            f.write(f"Accuracy: {self.results['accuracy']:.4f}\n")
            f.write(f"Precision: {self.results['precision']:.4f}\n")
            f.write(f"Recall: {self.results['recall']:.4f}\n")
            f.write(f"F1-Score: {self.results['f1_score']:.4f}\n")
        
        logging.info(f"\n✅ Results saved to: {results_path}")
    
    def plot_confusion_matrix(self):
        """Plot and save confusion matrix"""
        if not self.results or 'confusion_matrix' not in self.results:
            logging.error("❌ No confusion matrix available.")
            return
            
        # Create confusion matrix plot
        plt.figure(figsize=(8, 6))
        labels = ['No Defect', 'Defect']
        sns.heatmap(self.results['confusion_matrix'], 
                   annot=True, fmt='d', 
                   cmap='Blues',
                   xticklabels=labels,
                   yticklabels=labels)
        plt.title('Confusion Matrix', fontsize=16, fontweight='bold')
        plt.xlabel('Predicted', fontsize=12)
        plt.ylabel('Actual', fontsize=12)
        plt.tight_layout()
        
        # Save confusion matrix
        cm_path = os.path.join(os.path.dirname(os.path.dirname(self.model_path)), 
                              "results", "pipeline_confusion_matrix.png")
        plt.savefig(cm_path, dpi=300, bbox_inches='tight')
        logging.info(f"✅ Confusion matrix saved: {cm_path}")
    
    def plot_precision_recall_curve(self):
        """Plot precision-recall curve"""
        if not self.metrics:
            print("❌ No metrics available.")
            return
            
        plt.figure(figsize=(12, 8))
        for i, class_name in self.class_names.items():
            plt.plot(self.metrics.box.r[i], self.metrics.box.p[i], 
                    marker='o', label=f'{class_name}', linewidth=2)
        
        plt.xlabel('Recall', fontsize=12)
        plt.ylabel('Precision', fontsize=12)
        plt.title('Precision-Recall Curve', fontsize=16, fontweight='bold')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('/kaggle/working/precision_recall_curve.png', dpi=300, bbox_inches='tight')
        plt.show()
        print("✅ Precision-Recall curve saved: /kaggle/working/precision_recall_curve.png")
    
    def test_on_sample_images(self, num_samples=5):
        """Test model on sample images"""
        print(f"\n🔍 Testing on {num_samples} sample images...")
        
        # Get sample images from validation set
        val_images_path = '/kaggle/input/pipeline-defect-dataset/images/images/train/'
        sample_images = [f for f in os.listdir(val_images_path) if f.endswith(('.jpg', '.jpeg', '.png'))][:num_samples]
        
        results = []
        for img_name in sample_images:
            img_path = os.path.join(val_images_path, img_name)
            result = self.model.predict(img_path, conf=0.5, save=False)
            results.append((img_name, result))
            
            print(f"\n📷 {img_name}:")
            if len(result[0].boxes) > 0:
                for i, box in enumerate(result[0].boxes):
                    class_name = self.class_names[int(box.cls)]
                    confidence = float(box.conf)
                    print(f"   {i+1}. {class_name}: {confidence:.3f}")
            else:
                print("   No defects detected")
        
        return results
    

def main():
    """Main evaluation function"""
    # Configuration - use relative paths
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    MODEL_PATH = os.path.join(base_dir, "models", "pipeline_defect_detection.pt")
    DATA_CONFIG = os.path.join(base_dir, "datasets", "pipeline_dataset", "config.yaml")
    
    # Verify paths exist
    if not os.path.exists(MODEL_PATH):
        logging.error(f"Model not found: {MODEL_PATH}")
        sys.exit(1)
        
    if not os.path.exists(DATA_CONFIG):
        logging.error(f"Data config not found: {DATA_CONFIG}")
        sys.exit(1)
    
    logging.info(f"Using model: {MODEL_PATH}")
    logging.info(f"Using data config: {DATA_CONFIG}")
    
    # Initialize evaluator
    evaluator = ModelEvaluator(MODEL_PATH, DATA_CONFIG)
    
    # Load model
    evaluator.load_model()
    
    # Run evaluation
    evaluator.evaluate_model()
    
    # Print detailed results
    evaluator.print_detailed_metrics()
    
    # Generate confusion matrix
    evaluator.plot_confusion_matrix()
    
    logging.info("\n" + "="*60)
    logging.info("🎉 EVALUATION COMPLETED!")
    logging.info("="*60)

if __name__ == "__main__":
    main()