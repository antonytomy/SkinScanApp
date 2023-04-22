from flask import Flask,request, jsonify, render_template
import base64
import numpy as np
import io
from PIL import Image, ImageOps
import tensorflow as tf
import keras
from keras.models import load_model
from keras.models import Sequential


app = Flask(__name__)


@app.route("/")
def start():
    return render_template("index.html")

def get_model():
    global model
    model=load_model("keras_model.h5")
    print("Model Loaded!")



print("Loading Skin Lesion Classification Model")
get_model()

@app.route("/predict",methods=["POST"])
def predict():
    message=request.get_json(force=True)
    print(message)
    encoded=message["image"]
    decoded=base64.b64decode(encoded)
    image=Image.open(io.BytesIO(decoded)).convert("RGB")
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    size = (224, 224)
    image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)
    image_array = np.asarray(image)
    normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1
    data[0] = normalized_image_array
    prediction=np.argmax(model.predict(data))
   
    num_to_class_name={
        0 :"Actinic keratosis",
        1: "Basal cell carcinoma",
        2: "Benign keratosis-like lesions",
        3: "Dermatofibroma",
        4: "Melanocytic nevi",
        5: "Melanoma",
        6: "Vascular lesion"
    }
    response={
        "prediction":{
            "skin_class":num_to_class_name[prediction]
        }
    }

    return jsonify(response)
if __name__ =="__main__":
    app.run(debug=True)