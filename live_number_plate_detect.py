"""
Real-time License Plate Detection from Mobile Camera Stream
Detects license plates and performs OCR using YOLO and EasyOCR
"""

from ultralytics import YOLO
import cv2
import csv
import datetime
import easyocr
import sys

def main():
    # CONFIG
    model_path = r"best.pt"  # Update this path to your model
    url = "http://192.168.0.193:8080/video"  # Mobile camera stream URL
    
    # Load Model
    print("Loading YOLO model...")
    model = YOLO(model_path)
    
    # Initialize OCR reader
    print("Initializing OCR reader...")
    reader = easyocr.Reader(['en'])
    
    # Camera Stream
    print(f"Connecting to camera stream: {url}")
    cap = cv2.VideoCapture(url)
    if not cap.isOpened():
        print("Error: Mobile camera stream could not be opened")
        sys.exit(1)
    
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0:
        fps = 30
    
    # Video Writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter("number_plate_output.mp4", fourcc, fps, (frame_width, frame_height))
    
    # CSV File for logging
    csv_file = open("detected_plates.csv", mode="w", newline="")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(["Timestamp", "Plate_Number"])  # headers
    
    print("System started. Press 'Q' to stop.")
    
    # Loop
    frame_count = 0
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Frame read error - check connection")
                break
            
            # YOLO Detection
            results = model.predict(frame, conf=0.5, verbose=False)
            
            annotated_frame = results[0].plot()  # Draw bounding boxes
            
            # Extract detected plates
            for box in results[0].boxes:
                cls_id = int(box.cls[0])  # class ID
                label = model.names[cls_id]  # class name (e.g. "plate")
                
                # Crop detected plate region
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                plate_img = frame[y1:y2, x1:x2]
                
                # OCR (recognition of plate number)
                try:
                    ocr_result = reader.readtext(plate_img)
                    if len(ocr_result) > 0:
                        plate_text = ocr_result[0][-2]  # detected text
                        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        
                        # Save to CSV
                        csv_writer.writerow([timestamp, plate_text])
                        csv_file.flush()
                        print(f"Detected Plate: {plate_text} | Time: {timestamp}")
                except Exception as e:
                    print(f"OCR error: {e}")
            
            # Show live window
            cv2.imshow("YOLO Licence Plate Detection", annotated_frame)
            
            # Save video
            out.write(annotated_frame)
            
            frame_count += 1
            
            # Press Q to quit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("Stopping detection...")
                break
    
    finally:
        cap.release()
        out.release()
        csv_file.close()
        cv2.destroyAllWindows()
        print(f"\nDetection finished.")
        print(f"Frames processed: {frame_count}")
        print(f"Video saved as: number_plate_output.mp4")
        print(f"CSV saved as: detected_plates.csv")

if __name__ == "__main__":
    main()
