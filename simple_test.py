"""
Simplified License Plate Detection Testing Module
Tests YOLO model on static images without OCR
"""

from ultralytics import YOLO
import cv2
import torch
import pandas as pd
import os
import sys
from pathlib import Path

class LicensePlateDetector:
    def __init__(self, model_path, conf=0.25, iou=0.45):
        """Initialize detector with model and settings"""
        self.model_path = model_path
        self.conf = conf
        self.iou = iou
        self.device = 0 if torch.cuda.is_available() else "cpu"

        print("Loading YOLO model...")
        self.model = YOLO(model_path)

        self.detections = []

    def test_on_images(self, image_list, show=False, save=True):
        """Test model on static images"""
        print(f"\n{'='*60}")
        print(f"Testing on {len(image_list)} images")
        print(f"{'='*60}")

        results = self.model.predict(source=image_list, show=show, save=save, conf=self.conf)

        for result in results:
            print(f"\nImage: {result.path}")
            if result.boxes:
                for box in result.boxes:
                    label = self.model.names[int(box.cls)]
                    confidence = float(box.conf)
                    print(f"  - Label: {label}, Confidence: {confidence:.4f}")
            else:
                print("  - No detections found")

        return results

    def check_libraries(self):
        """Check if required libraries are installed"""
        libraries = ["ultralytics", "cv2", "pandas", "torch"]

        print(f"\n{'='*60}")
        print("Checking installed libraries")
        print(f"{'='*60}")

        for lib in libraries:
            try:
                __import__(lib)
                print(f"✅ {lib} is installed")
            except ImportError:
                print(f"❌ {lib} is NOT installed")

def main():
    # Settings
    model_path = "best.pt"  # Update this to your model path

    # Check if model exists
    if not os.path.exists(model_path):
        print(f"Error: Model file '{model_path}' not found")
        sys.exit(1)

    # Initialize detector
    detector = LicensePlateDetector(model_path)

    # Check libraries
    detector.check_libraries()

    # Example 1: Test on images
    image_list = [
        r"License Number Plates Dataset\Cars\DSC_1056.JPG",
        r"License Number Plates Dataset\Cars\DSC_1090.JPG"
    ]

    # Filter to only existing images
    image_list = [img for img in image_list if os.path.exists(img)]

    if image_list:
        print("\n--- Testing on images ---")
        results = detector.test_on_images(image_list, show=False, save=False)
    else:
        print("\nNo test images found. Please check the paths.")

    print("\nTesting complete!")

if __name__ == "__main__":
    main()