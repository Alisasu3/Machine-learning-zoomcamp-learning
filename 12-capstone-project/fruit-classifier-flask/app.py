from flask import Flask, request, jsonify
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.applications.xception import preprocess_input
from PIL import Image
import io
import os


MODEL_PATH = os.path.join("model", "xception_fruits_final.h5")
model = keras.models.load_model(MODEL_PATH)

class_names = [
    "Apple Golden 1",
    "Banana",
    "Mango",
    "Orange",
    "Strawberry"
]

app = Flask(__name__)

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict")
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file provided. Use form-data key 'file'."}), 400

    file = request.files["file"]

    # Load and preprocess image
    img = Image.open(io.BytesIO(file.read())).convert("RGB")
    img = img.resize((150, 150))

    x = np.array(img)
    X = np.expand_dims(x, axis=0)
    X = preprocess_input(X)

    # Predict (logits -> probs)
    logits = model.predict(X, verbose=0)
    probs = tf.nn.softmax(logits).numpy()[0]

    pred_idx = int(np.argmax(probs))
    pred_class = class_names[pred_idx]
    confidence = float(probs[pred_idx])

    return jsonify({
        "prediction": pred_class,
        "confidence": round(confidence, 4)
    })

if __name__ == "__main__":
    # host=0.0.0.0 makes it reachable in Docker
    app.run(host="0.0.0.0", port=5000, debug=True)
