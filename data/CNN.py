from tensorflow.keras.applications import ResNet50, VGG16
from tensorflow.keras.models import load_model
from tensorflow.saved_model import LoadOptions


def get_model(model_name, model_path):
    if model_name:
        if model_name == "VGG16":
            return VGG16(weights="imagenet")
        elif model_name == "ResNet50":
            return ResNet50(weights="imagenet")
    else:
        model = load_model(model_path)
    return model
