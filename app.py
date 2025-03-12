import os
import torch
import cv2
import numpy as np
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from PIL import Image

# Define paths
UPLOAD_FOLDER = "uploads"
OUTPUT_FOLDER = "outputs"

# Ensure directories exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Load YOLOv5 model (force CPU, use smallest model)
try:
    model = torch.hub.load("ultralytics/yolov5", "yolov5n", device="cpu")  # 🔹 Use 'yolov5n' (nano version)
    model.conf = 0.5  # 🔹 Reduce confidence threshold to process fewer objects
    print("✅ YOLOv5 nano model loaded successfully.")
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
        # Convert image to RGB format & resize to reduce memory usage
        image = Image.open(file_path).convert("RGB")
        image = image.resize((320, 320))  # 🔹 Resize to 320x320 (smaller input, lower memory use)

        # Run YOLOv5 inference
        results = model(image)

        # Convert to NumPy array for OpenCV
        img = np.array(image)

        # Draw bounding boxes (limit processing to 5 objects max)
        for i, (*box, conf, cls) in enumerate(results.xyxy[0]):
            if i >= 5:  # 🔹 Process max 5 objects to save memory
                break
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
    port = int(os.environ.get("PORT", 10000))  # Default to 10000 if PORT is not set
    app.run(host="0.0.0.0", port=port, debug=True)
