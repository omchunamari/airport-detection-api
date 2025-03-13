from flask import Flask, request, jsonify, send_file
import torch
import os
import ssl
from PIL import Image
import io
from flask_cors import CORS  # Import CORS for frontend integration

app = Flask(__name__)
CORS(app)  # Enable CORS to allow frontend to communicate

# 🔥 Disable SSL Verification for Torch Hub (Fixes SSL errors)
ssl._create_default_https_context = ssl._create_unverified_context

# ✅ Load YOLOv5 model safely
MODEL_PATH = "/Users/omchunamari/Desktop/Final Year Project/yolov5best.pt"
model = torch.hub.load(
    "ultralytics/yolov5",
    "custom",
    path=MODEL_PATH,
    force_reload=True,
    trust_repo=True,  # Avoid SSL warnings
)

@app.route("/detect", methods=["POST"])
def detect():
    if "image" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["image"]
    image = Image.open(file.stream).convert("RGB")  # ✅ Convert image to RGB

    # 🔥 Run YOLOv5 inference
    results = model(image)

    # 🖼️ Render bounding boxes
    results.render()
    
    # ✅ Convert the image correctly before sending
    img_bytes = io.BytesIO()
    Image.fromarray(results.ims[0]).save(img_bytes, format="JPEG")
    img_bytes.seek(0)

    return send_file(img_bytes, mimetype="image/jpeg")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=4000, debug=True)  # ✅ Allow external access
