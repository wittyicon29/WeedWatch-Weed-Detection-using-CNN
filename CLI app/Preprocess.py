import cv2
import numpy as np

# Function to preprocess the input image
def preprocess_image(image_path):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (512, 512))  # Resize the image to the required input shape
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img / 255.0  # Normalize the pixel values to the range [0, 1]
    return img[np.newaxis, ...]
