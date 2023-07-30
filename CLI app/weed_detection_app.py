from Inference import perform_inference
from Model import load_model
from Prediction import display_results


def main():
    model_path = "C:\\Users\\Atharva\\Downloads\\VS\\CLI app\\trained-model.h5"
    model = load_model(model_path)

    import sys
    if len(sys.argv) != 2:
        print("Usage: python weed_detection_app.py path/to/your/image.jpg")
        return
    image_path = sys.argv[1]

    # Perform inference and display results
    predictions = perform_inference(model, image_path)
    display_results(predictions[0], image_path)

if __name__ == "__main__":
    main()
