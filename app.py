import os
import subprocess

# Automatically install required dependencies
required_packages = ["flask", "flask-cors", "pillow", "torch", "torchvision", "ultralytics", "opencv-python","gitpython"]

for package in required_packages:
    try:
        __import__(package.replace("-", "_"))  # Import package to check if it's installed
    except ImportError:
        print(f"Installing {package}...")
        subprocess.check_call(["pip", "install", package])

# Now import the required libraries
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import cv2
import torch
from PIL import Image
import numpy as np

# Define paths
UPLOAD_FOLDER = "uploads"
OUTPUT_FOLDER = "outputs"
MODEL_PATH = "yolov5best.pt"  # Change this to the actual path of best.pt

# Ensure directories exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Load custom YOLOv5 model
model = torch.hub.load("ultralytics/yolov5", "custom", path=MODEL_PATH, force_reload=True)

app = Flask(__name__)
CORS(app)  # Enable CORS for frontend communication

@app.route("/upload/", methods=["POST"])
def upload_file():
    if "file" not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files["file"
    ]
    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400

    file_path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(file_path)

    # Run YOLOv5 inference on uploaded image
    image = Image.open(file_path)
    results = model(image)

    # Draw bounding boxes
    img = np.array(image)
    for *box, conf, cls in results.xyxy[0]:  # Iterate over detections
        x1, y1, x2, y2 = map(int, box)
        label = f"{model.names[int(cls)]} {conf:.2f}"
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    output_path = os.path.join(OUTPUT_FOLDER, file.filename)
    Image.fromarray(img).save(output_path)

    return jsonify({"filename": file.filename})

@app.route("/outputs/<filename>")
def get_output_image(filename):
    return send_from_directory(OUTPUT_FOLDER, filename)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)
