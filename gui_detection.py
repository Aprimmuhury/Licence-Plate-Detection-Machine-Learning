#!/usr/bin/env python
"""
Web-based License Plate Detection UI
A Flask application with CSS styling for image upload and YOLO inference.
"""

import os
import uuid
from pathlib import Path
from flask import Flask, render_template, request, url_for
from werkzeug.utils import secure_filename
from ultralytics import YOLO
import cv2

BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR / "uploads"
RESULT_DIR = BASE_DIR / "static" / "results"
DEFAULT_MODEL = BASE_DIR / "best.pt"

UPLOAD_DIR.mkdir(exist_ok=True)
RESULT_DIR.mkdir(parents=True, exist_ok=True)

app = Flask(__name__, static_folder="static", template_folder="templates")
app.config["MAX_CONTENT_LENGTH"] = 12 * 1024 * 1024
app.config["UPLOAD_FOLDER"] = str(UPLOAD_DIR)
app.config["RESULT_FOLDER"] = str(RESULT_DIR)


def load_model(model_path: str):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    return YOLO(model_path)


@app.route("/", methods=["GET", "POST"])
def index():
    status_message = None
    detection_info = None
    image_url = None
    model_path = str(DEFAULT_MODEL)
    confidence = 0.25

    if request.method == "POST":
        confidence = float(request.form.get("confidence", "0.25"))
        uploaded_file = request.files.get("image")

        if not uploaded_file or uploaded_file.filename == "":
            status_message = "Please select an image to upload."
            return render_template("index.html", image_url=image_url, status_message=status_message,
                                   detection_info=detection_info, model_path=model_path, confidence=confidence)

        filename = secure_filename(uploaded_file.filename)
        image_path = UPLOAD_DIR / f"{uuid.uuid4().hex}_{filename}"
        uploaded_file.save(image_path)

        try:
            model = load_model(model_path)
            results = model.predict(source=str(image_path), conf=confidence, verbose=False)
            result = results[0]
            output_image = result.plot()
            output_filename = f"result_{uuid.uuid4().hex}.jpg"
            output_path = RESULT_DIR / output_filename
            cv2.imwrite(str(output_path), output_image)
            image_url = url_for("static", filename=f"results/{output_filename}")

            if result.boxes is not None and len(result.boxes) > 0:
                lines = []
                for idx, box in enumerate(result.boxes):
                    cls_id = int(box.cls[0])
                    label = model.names[cls_id]
                    conf_score = float(box.conf[0])
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    lines.append(f"{idx + 1}. {label} ({conf_score:.2f}) at [{x1}, {y1}, {x2}, {y2}]")
                detection_info = "\n".join(lines)
            else:
                detection_info = "No detections found."

            status_message = "Detection complete! Scroll down to view the results."
        except Exception as error:
            status_message = f"Model error: {error}"

    return render_template("index.html", image_url=image_url, status_message=status_message,
                           detection_info=detection_info, model_path=model_path, confidence=confidence)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
