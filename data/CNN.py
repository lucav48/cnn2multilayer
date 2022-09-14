from tensorflow.keras.applications import *
from tensorflow.keras.models import load_model


def get_model(model_name, model_path, dataset):
    if model_path:
        model = load_model(model_path)
    else:
        if dataset.lower() == "cifar10":
            model = eval(model_name + "(weights='imagenet', input_shape=(32,32,3), classes=10, include_top=False)")
        elif dataset.lower() == "cifar100":
            model = eval(model_name + "(weights='imagenet', input_shape=(32,32,3), classes=100, include_top=False)")
        else:
            model = eval(model_name + "(weights='imagenet')")
    return model
