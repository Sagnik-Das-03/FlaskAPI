# model/mnist_model.py

import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np

# Load or define your MNIST model
def load_mnist_model(model_path):
    model = load_model(model_path)
    return model

# Predict digit from image
def predict_digit(model, image):
    image = np.expand_dims(image, axis=0)
    image = np.expand_dims(image, axis=-1)
    image = image.astype('float32') / 255.0
    prediction = model.predict(image)
    return np.argmax(prediction, axis=1)[0]
