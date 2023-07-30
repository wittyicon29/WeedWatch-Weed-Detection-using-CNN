from Preprocess import preprocess_image


def perform_inference(model, image_path):
    input_image = preprocess_image(image_path)
    predictions = model.predict(input_image)
    return predictions