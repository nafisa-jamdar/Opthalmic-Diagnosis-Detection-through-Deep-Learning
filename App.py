import cv2
import numpy as np
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog
import numpy as np
from keras.preprocessing import image
from keras.models import Sequential
from keras.layers import Dense
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog
import numpy as np
from keras.preprocessing import image
from keras.models import Sequential
from keras.layers import Dense
from keras.models import model_from_json
import tensorflow as tf
from flask import Flask, render_template, request, send_from_directory

app = Flask(__name__)

UPLOAD_FOLDER = "uploads"
STATIC_FOLDER = "static"

# Load the model
json_file = open('model_dense.json', 'r')
loaded_model_json = json_file.read()
json_file.close()

# Replace any occurrence of 'softmax_v2' with 'softmax'
loaded_model_json = loaded_model_json.replace('softmax_v2', 'softmax')

# Create the model from JSON
model = model_from_json(loaded_model_json)

# Load weights into the model
model.load_weights("model_dense.weights.h5")  # Ensure the path to weights file is correct

IMAGE_SIZE = 64

# Preprocess an image
def preprocess_image(image):
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, [IMAGE_SIZE, IMAGE_SIZE])
    image /= 255.0  # normalize to [0,1] range

    return image

# Read the image from path and preprocess
def load_and_preprocess_image(path):
    image = tf.io.read_file(path)
    return preprocess_image(image)

# Predict & classify image
def classify(model, image_path):
    preprocessed_image = load_and_preprocess_image(image_path)
    preprocessed_image = tf.reshape(preprocessed_image, (1, IMAGE_SIZE, IMAGE_SIZE, 3))

    prob = model.predict(preprocessed_image)[0]
    print(prob)

    # Get the index of the maximum probability
    predicted_label_index = np.argmax(prob)

    # Mapping index to label name
    label_names = ['glaucoma',"normal","cataract","diabetic_retinopathy",]
    # Replace with your actual label names

    label = label_names[predicted_label_index]
    classified_prob = prob[predicted_label_index]

    return label, classified_prob

# Home page
@app.route("/")
def home():
    return render_template("home.html")

@app.route("/classify", methods=["POST", "GET"])
def upload_file():
    if request.method == "GET":
        return render_template("home.html")
    else:
        file = request.files["image"]
        upload_image_path = os.path.join(UPLOAD_FOLDER, file.filename)
        print(upload_image_path)
        file.save(upload_image_path)

        label, prob = classify(model, upload_image_path)
        prob = round((prob * 100), 2)

    return render_template(
        "classify.html", image_file_name=file.filename, label=label, prob=prob
    )

@app.route("/classify/<filename>")
def send_file(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)

if __name__ == "__main__":
    app.run()
