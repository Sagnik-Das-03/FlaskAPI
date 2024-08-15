from flask import Flask, jsonify, request
from model.mnist_model import load_mnist_model, predict_digit
import numpy as np
from PIL import Image
import io
import tensorflow as tf
app = Flask(__name__)

# Load the MNIST model
model_path = "handwritten_digits.h5"  # Replace with the actual path
model = tf.keras.models.load_model(model_path)
@app.route('/')
def home():
    return 'Hello, World!'

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400

    file = request.files['file']
    image = Image.open(file).convert('L').resize((28, 28))  # Convert to grayscale and resize
    image = np.array(image)
    digit = predict_digit(model, image)

    return jsonify({'digit': int(digit)})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8088)
