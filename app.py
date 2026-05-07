from flask import Flask, request, send_file, jsonify
from ultralytics import YOLO
import numpy as np
import cv2
from datetime import datetime
import os
from PIL import Image
import io
from file_utils import ensure_directories, saveFiles

from config import MODEL_PATH, CONF_THRESHOLD

app = Flask(__name__)

ensure_directories()

model = YOLO(MODEL_PATH, task='detect')

@app.route("/infer/json", methods=["POST"])
def infer_json():
    try:
        # 📥 receive raw JPEG bytes
        image_bytes = request.data

        _, results = run_inference(image_bytes)
         
        detections = []

        for box in results.boxes:

            x1, y1, x2, y2 = box.xyxy[0].tolist()
            conf = float(box.conf[0])
            cls = int(box.cls[0])

            detections.append({
                "class": model.names[cls],
                "confidence": conf,
                "bbox": [x1, y1, x2, y2]
            })

        return jsonify({
            "detections": detections
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/infer/image", methods=["POST"])
def infer_image():

    try:
        image_bytes = request.data

        _, results = run_inference(image_bytes)

        # Draw boxes
        plotted = results.plot()
        # BGR -> RGB
        plotted_rgb = cv2.cvtColor(
            plotted,
            cv2.COLOR_BGR2RGB
        )

        # numpy -> PIL
        output_image = Image.fromarray(plotted_rgb)

        # save to memory
        img_io = io.BytesIO()
        output_image.save(img_io, format="JPEG")
        img_io.seek(0)

        return send_file(
            img_io,
            mimetype="image/jpeg"
        )

    except Exception as e:
        return jsonify({
            "error": str(e)
        }), 500

@app.route("/test", methods=["GET", "POST"])
def test():
    print("i was called")
    print(request.data)  # optional debug
    return jsonify({"status": "ok"})

@app.route("/test_res", methods=["POST"])
def test_res():
    print("called")
    try:
        # 📥 receive raw JPEG bytes
        image_bytes = request.data

        # convert to numpy
        np_arr = np.frombuffer(image_bytes, np.uint8)

        # decode image
        image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        
        if image is not None:
            print(image.shape)
        else:
            print("Image decoding failed")
        return jsonify({"status": "ok"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

def run_inference(image_bytes):

    # bytes -> PIL image
    raw_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    # inference
    #results = model.predict(source=image, conf=CONF_THRESHOLD, imgsz=1280)[0]

    results = model.predict(
        source=raw_image,
        conf=CONF_THRESHOLD
    )[0]

    #saveFiles(image_bytes,raw_image,results.plot())

    return raw_image, results



if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)