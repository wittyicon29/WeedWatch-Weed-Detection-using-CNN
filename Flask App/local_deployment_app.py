from flask import Flask, request, jsonify
import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load the trained model
model = load_model("C:\\Users\\Atharva\\Downloads\\VS\\Weed Detection\\trained-model.h5")

# Create a Flask app
app = Flask(__name__)

# Define a route for prediction API endpoint
@app.route("/predict", methods=["POST"])
def predict():
    # Get the image from the request
    image = request.files["image"]

    # Preprocess the image
    image = cv2.imread(image)
    image = cv2.resize(image, (512, 512))  # Resize to match the input size of your model
    image = image.astype(np.float32) / 255.0  # Normalize pixel values
    image = np.expand_dims(image, axis=0)  # Add batch dimension

    # Perform prediction
    prediction = model.predict(image)

    # Process the prediction result
    result = {
        "class_0_probability": float(prediction[0][0]),
        "class_1_probability": float(prediction[0][1])
    }

    # Return the prediction result as JSON response
    return jsonify(result)

# Run the Flask app
#if __name__ == "__main__":
    #app.run()
