"""
YOLOv8 License Plate Detection Model Training
Trains a YOLO model on license plate dataset
"""

from ultralytics import YOLO
import pandas as pd
import os
import sys
from pathlib import Path

def train_model(data_yaml_path, model_size="n", epochs=50, imgsz=640, 
                device=None, batch_size=16):
    """
    Train YOLOv8 model for license plate detection
    
    Args:
        data_yaml_path: Path to data.yaml file
        model_size: YOLOv8 model size (n, s, m, l, x)
        epochs: Number of training epochs
        imgsz: Image size for training
        device: GPU device ID (default: 0 if available, else cpu)
        batch_size: Batch size for training
    """
    
    print(f"{'='*60}")
    print("YOLOv8 License Plate Training")
    print(f"{'='*60}")
    
    # Check if data.yaml exists
    if not os.path.exists(data_yaml_path):
        print(f"Error: data.yaml not found at {data_yaml_path}")
        sys.exit(1)
    
    print(f"\nDataset config: {data_yaml_path}")
    print(f"Model size: yolov8{model_size}")
    print(f"Epochs: {epochs}")
    print(f"Image size: {imgsz}x{imgsz}")
    print(f"Batch size: {batch_size}")
    
    # Load YOLOv8 model
    print(f"\nLoading yolov8{model_size}.pt model...")
    model = YOLO(f"yolov8{model_size}.pt")
    
    # Train the model
    print("\nStarting training...")
    results = model.train(
        data=data_yaml_path,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch_size,
        device=device,
        patience=10,  # Early stopping patience
        save=True,
        name="license_plate_detector"
    )
    
    print("\n" + "="*60)
    print("Training completed!")
    print("="*60)
    
    # Display final results
    try:
        results_csv = "runs/detect/license_plate_detector/results.csv"
        if os.path.exists(results_csv):
            df = pd.read_csv(results_csv)
            print("\nFinal training metrics:")
            print(df.tail(1).to_string())
        
        # Model locations
        best_model = "runs/detect/license_plate_detector/weights/best.pt"
        last_model = "runs/detect/license_plate_detector/weights/last.pt"
        
        if os.path.exists(best_model):
            print(f"\n✅ Best model saved: {best_model}")
        if os.path.exists(last_model):
            print(f"✅ Last model saved: {last_model}")
    
    except Exception as e:
        print(f"Error reading results: {e}")
    
    return results

def validate_model(model_path, data_yaml_path, imgsz=640):
    """
    Validate trained model
    
    Args:
        model_path: Path to trained model (.pt file)
        data_yaml_path: Path to data.yaml file
        imgsz: Image size for validation
    """
    
    print(f"\n{'='*60}")
    print("Model Validation")
    print(f"{'='*60}")
    
    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}")
        return None
    
    print(f"Loading model: {model_path}")
    model = YOLO(model_path)
    
    print("Running validation...")
    metrics = model.val(data=data_yaml_path, imgsz=imgsz)
    
    return metrics

def test_model_on_image(model_path, image_path, conf=0.25):
    """
    Test model on single image
    
    Args:
        model_path: Path to trained model
        image_path: Path to image
        conf: Confidence threshold
    """
    
    print(f"\n{'='*60}")
    print("Testing on Image")
    print(f"{'='*60}")
    
    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}")
        return None
    
    if not os.path.exists(image_path):
        print(f"Error: Image not found at {image_path}")
        return None
    
    print(f"Loading model: {model_path}")
    model = YOLO(model_path)
    
    print(f"Running inference on: {image_path}")
    results = model.predict(source=image_path, conf=conf, save=True, save_txt=True)
    
    print(f"Predictions saved")
    return results

def main():
    """Main training pipeline"""
    
    # Settings
    DATA_YAML = "number detection Project.v2i.yolov8/data.yaml"  # Update this path
    MODEL_SIZE = "n"  # nano (n), small (s), medium (m), large (l), xlarge (x)
    EPOCHS = 50
    IMGSZ = 640
    BATCH_SIZE = 16
    DEVICE = "cpu"  # Use CPU since CUDA is not available
    
    # Check if data.yaml exists
    if not os.path.exists(DATA_YAML):
        print(f"Error: {DATA_YAML} not found")
        print("Please update DATA_YAML path to point to your dataset config")
        return
    
    # Train model
    print("Step 1: Training model...")
    train_model(
        data_yaml_path=DATA_YAML,
        model_size=MODEL_SIZE,
        epochs=EPOCHS,
        imgsz=IMGSZ,
        device=DEVICE,
        batch_size=BATCH_SIZE
    )
    
    # Validate model
    print("\n\nStep 2: Validating model...")
    best_model = "runs/detect/license_plate_detector/weights/best.pt"
    if os.path.exists(best_model):
        validate_model(best_model, DATA_YAML, imgsz=IMGSZ)
    
    # Test on sample image (optional)
    print("\n\nStep 3: Testing on sample image (optional)...")
    sample_image = "License Number Plates Dataset/Cars/DSC_1056.JPG"
    if os.path.exists(sample_image) and os.path.exists(best_model):
        test_model_on_image(best_model, sample_image)
    
    print("\n" + "="*60)
    print("Training pipeline complete!")
    print("="*60)

if __name__ == "__main__":
    main()
