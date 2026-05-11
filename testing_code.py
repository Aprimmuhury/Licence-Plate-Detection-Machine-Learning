"""
License Plate Detection Testing Module
Tests YOLO model on static images, videos, and streams
Performs OCR on detected plates
"""

from ultralytics import YOLO
import cv2
import torch
import pandas as pd
import easyocr
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
        
        print("Initializing OCR reader...")
        self.reader = easyocr.Reader(['en'])
        
        self.detections = []
    
    def test_on_images(self, image_list, show=True, save=True):
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
        
        return results
    
    def test_on_video(self, video_path, output_path="output_annotated.mp4", save_csv=True):
        """Test model on video file"""
        print(f"\n{'='*60}")
        print(f"Testing on video: {video_path}")
        print(f"{'='*60}")
        
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            print(f"Error: Could not open video {video_path}")
            return None
        
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))
        
        rows = []
        frame_idx = 0
        detected_count = 0
        
        print(f"Processing video... (Press Ctrl+C to stop)")
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # YOLO prediction
                res = self.model.predict(frame, conf=self.conf, iou=self.iou, 
                                        device=self.device, verbose=False)[0]
                
                if res.boxes is not None and len(res.boxes) > 0:
                    boxes = res.boxes.xyxy.cpu().numpy()
                    scores = res.boxes.conf.cpu().numpy()
                    
                    for i, (x1, y1, x2, y2) in enumerate(boxes):
                        conf_score = float(scores[i])
                        detected_count += 1
                        
                        # Crop plate area for OCR
                        plate_crop = frame[int(y1):int(y2), int(x1):int(x2)]
                        plate_text = ""
                        
                        if plate_crop.size > 0:
                            try:
                                result = self.reader.readtext(plate_crop)
                                if result:
                                    plate_text = result[0][1]  # OCR text
                            except Exception as e:
                                print(f"OCR error on frame {frame_idx}: {e}")
                        
                        # Draw bounding box + plate number
                        label = f"{plate_text} {conf_score:.2f}" if plate_text else f"{conf_score:.2f}"
                        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                        cv2.putText(frame, label, (int(x1), max(0, int(y1) - 5)), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                        
                        # Log to CSV
                        rows.append({
                            "frame": frame_idx,
                            "x1": float(x1), "y1": float(y1),
                            "x2": float(x2), "y2": float(y2),
                            "conf": conf_score,
                            "plate_text": plate_text
                        })
                
                out.write(frame)
                frame_idx += 1
                
                if frame_idx % 30 == 0:
                    print(f"  Processed {frame_idx} frames, {detected_count} detections so far...")
        
        finally:
            cap.release()
            out.release()
            
            # Save CSV if requested
            if save_csv and rows:
                csv_path = output_path.replace(".mp4", "_detections.csv")
                pd.DataFrame(rows).to_csv(csv_path, index=False)
                print(f"\n✅ Video saved: {output_path}")
                print(f"✅ Detections CSV saved: {csv_path}")
                print(f"Total detections: {detected_count}")
        
        return output_path
    
    def check_libraries(self):
        """Check if required libraries are installed"""
        libraries = ["ultralytics", "easyocr", "cv2", "pandas", "torch"]
        
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
    
    # Example 2: Test on video
    video_path = "pictures.mp4"  # Update this to your video path
    if os.path.exists(video_path):
        print("\n--- Testing on video ---")
        detector.test_on_video(video_path, output_path="output_annotated.mp4")
    else:
        print(f"\nNote: Video file '{video_path}' not found for testing")
    
    print("\nTesting complete!")

if __name__ == "__main__":
    main()
