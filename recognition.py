from tensorflow import keras
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions

# Load the pre-trained model
model = ResNet50(weights='imagenet')

def recognize_image(img_path):
    """
    Recognize the objects in the image.
    :param img_path: str, the path to the image
    :return: list, containing the top predictions
    """
    # Load the image
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = preprocess_input(x)

    # Make predictions
    predictions = model.predict(x)

    # Decode the predictions
    labels = decode_predictions(predictions, top=3)

    return labels

# Example usage
img_path = "example.jpg"
predictions = recognize_image(img_path)
for label in predictions[0]:
    print(f"{label[1]}: {label[2]:.2f}%")
