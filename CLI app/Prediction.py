import numpy as np

class_labels = ["weed", "non-weed"]

# Function to display the results for a single image prediction
def display_results(prediction, file_name):
    predicted_class_index = np.argmax(prediction)
    predicted_class = class_labels[predicted_class_index]

    print("File:", file_name)
    print("Predicted Class:", predicted_class)
    print("Prediction Probabilities:", prediction)
