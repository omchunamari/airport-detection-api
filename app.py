import os
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import cv2
import torch
from PIL import Image
import numpy as np

# Define paths
UPLOAD_FOLDER = "uploads"
OUTPUT_FOLDER = "outputs"
MODEL_PATH = "yolov5best.pt"  # Ensure this file exists in your root directory

# Ensure directories exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Load GitHub Token from environment (set in Render)
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")

# Set GitHub token in environment (required for torch.hub)
if GITHUB_TOKEN:
    os.environ["GITHUB_TOKEN"] = GITHUB_TOKEN

# Load YOLOv5 model
try:
    model = torch.hub.load("ultralytics/yolov5", "custom", path=MODEL_PATH, force_reload=True, trust_repo=True)
    print("✅ YOLOv5 model loaded successfully.")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    model = None  # Prevent further errors

app = Flask(__name__)

# ✅ Allow CORS only for Vercel frontend
CORS(app, resources={r"/upload/*": {"origins": "https://airport-object-detection.vercel.app"}})

@app.route("/")
def home():
    return jsonify({"message": "Airport Object Detection API is running!"})

@app.route("/upload/", methods=["POST"])
def upload_file():
    if "file" not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400

    file_path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(file_path)

    if model is None:
        return jsonify({"error": "Model not loaded. Try again later."}), 500

    try:
        # Convert image to RGB format
        image = Image.open(file_path).convert("RGB")
        results = model(image)

        # Convert to NumPy array for OpenCV
        img = np.array(image)

        # Draw bounding boxes
        for *box, conf, cls in results.xyxy[0]:
            x1, y1, x2, y2 = map(int, box)
            label = f"{model.names[int(cls)]} {conf:.2f}"
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Save output image
        output_path = os.path.join(OUTPUT_FOLDER, file.filename)
        Image.fromarray(img).save(output_path)

        return jsonify({"filename": file.filename})

    except Exception as e:
        return jsonify({"error": f"Processing failed: {e}"}), 500

@app.route("/outputs/<filename>")
def get_output_image(filename):
    return send_from_directory(OUTPUT_FOLDER, filename)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
