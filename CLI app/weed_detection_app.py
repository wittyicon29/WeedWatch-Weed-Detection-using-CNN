from Inference import perform_inference
from Model import load_model
from Prediction import display_results


def main():
    model_path = "path/to/your/model.h5"  # Replace with the path to your model file
    model = load_model(model_path)

    image_path = "path/to/your/image.jpg"  # Replace with the path to your input image
    predictions = perform_inference(model, image_path)

    display_results(predictions[0], image_path)

if __name__ == "__main__":
    main()
