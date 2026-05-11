# License Plate Detection Project

A YOLOv8-based license plate detection system with OCR capabilities for Bangladeshi and Pakistani vehicles.

## Project Structure

```
├── training_code.py              # Model training script
├── testing_code.py               # Model testing and inference script
├── live_number_plate_detect.py   # Live camera stream detection
├── requirements.txt              # Python dependencies
├── best.pt                       # Trained YOLO model weights
├── detected_plates.csv           # CSV log of detected plates
└── License Number Plates Dataset/# Training dataset
    ├── Cars/
    └── Plates/
```

## Installation

### 1. Prerequisites
- Python 3.8 or higher
- pip (Python package manager)
- (Optional) NVIDIA GPU with CUDA support for faster training

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Download Model Weights (if not included)

The `best.pt` file contains pre-trained weights. If you need to train from scratch:

```bash
python training_code.py
```

## Usage

### Training the Model

```bash
python training_code.py
```

Configuration options in `training_code.py`:
- `MODEL_SIZE`: Model size (n=nano, s=small, m=medium, l=large, x=xlarge)
- `EPOCHS`: Number of training epochs
- `BATCH_SIZE`: Batch size for training
- `IMGSZ`: Image size (default: 640)

### Testing/Inference

```bash
python testing_code.py
```

Features:
- Test on static images
- Test on video files
- Automatic OCR on detected plates
- CSV output with detections

### Live Camera Stream Detection

```bash
python live_number_plate_detect.py
```

Configuration in the script:
- `model_path`: Path to trained model
- `url`: Mobile camera stream URL (IP camera)
- Outputs:
  - `number_plate_output.mp4`: Annotated video
  - `detected_plates.csv`: Detected plate numbers with timestamps

### Web GUI Detection with CSS

```bash
python gui_detection.py
```

Open your browser at `http://127.0.0.1:5000/`.

Features:
- Stylish web interface built with Flask
- Upload an image file for inference
- Set detection confidence threshold
- View annotated results in the browser
- Download or save the result image from the server

## Features

✅ Real-time license plate detection using YOLOv8
✅ Optical Character Recognition (OCR) using EasyOCR
✅ Support for live camera streams
✅ Video processing and annotation
✅ CSV logging of detections
✅ GPU acceleration support
✅ Batch image processing

## Dataset Structure

Expected `data.yaml` format:

```yaml
path: /path/to/dataset
train: train/images
val: val/images

nc: 1
names: ['license_plate']
```

## Configuration

### For Live Camera Stream

Edit `live_number_plate_detect.py`:
```python
model_path = r"path/to/best.pt"
url = "http://your-camera-ip:8080/video"  # Your camera URL
```

### For Video Testing

Edit `testing_code.py`:
```python
video_path = "your_video.mp4"
```

## Output Files

- **number_plate_output.mp4**: Annotated video with bounding boxes
- **detected_plates.csv**: Log file with format:
  ```
  Timestamp,Plate_Number
  2024-01-15 10:30:45,AB-1234
  2024-01-15 10:31:12,XY-5678
  ```
- **output_annotated.mp4**: Video with detection results
- **output_annotated_detections.csv**: Detailed detection data with coordinates

## System Requirements

- **Minimum**: CPU, 4GB RAM, Python 3.8+
- **Recommended**: NVIDIA GPU (RTX series), 8GB+ RAM, CUDA 11.8+

## Troubleshooting

### Camera Connection Issues
- Verify camera URL is correct
- Ensure device is on the same network
- Test connection with: `python -c "import cv2; cv2.VideoCapture('YOUR_URL').isOpened()"`

### OCR Not Working
- Ensure EasyOCR is installed: `pip install easyocr`
- First run downloads language models (may take time)
- Check image quality and lighting

### CUDA/GPU Issues
- Verify NVIDIA drivers are installed
- Check CUDA availability: `python -c "import torch; print(torch.cuda.is_available())"`
- If GPU not available, falls back to CPU automatically

### Memory Issues
- Reduce `batch_size` in training
- Reduce `IMGSZ` value
- Close other applications

## Performance Tips

1. **Faster Inference**: Use smaller model size (nano/small)
2. **Better Accuracy**: Use larger model (medium/large)
3. **GPU Acceleration**: Ensure CUDA is properly installed
4. **Batch Processing**: Process multiple images/videos for efficiency

## License

This project uses YOLOv8 (by Ultralytics) licensed under AGPL-3.0 and EasyOCR (open-source).

## Support

For issues or questions:
1. Check the troubleshooting section
2. Review YOLOv8 documentation: https://docs.ultralytics.com/
3. Check EasyOCR documentation: https://github.com/JaidedAI/EasyOCR
