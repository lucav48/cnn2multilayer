from tensorflow.keras.applications import *
from tensorflow.keras.models import load_model


def get_model(model_name, model_path):
    if model_path:
        model = load_model(model_path)
    else:
        model = eval(model_name + "(weights='imagenet')")
    return model
