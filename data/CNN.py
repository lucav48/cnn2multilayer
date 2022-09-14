from tensorflow.keras.applications import *
from tensorflow.keras.models import load_model
import gdown
import os


def get_model(model_name, model_path, dataset):
    if model_path:
        model = load_model(model_path)
    else:
        # custom model for MNIST based on VGG16
        if dataset.lower() == "mnist":
            if not os.path.exists("data/mnist/vgg16-mnist.h5"):
                url = 'https://drive.google.com/uc?id=1ziOvcnj973dWlLcG2E7MfqW-RQ18sq5G'
                output = "data/mnist/vgg16-mnist.h5"
                gdown.download(url, output, quiet=False)
            model = load_model("data/mnist/vgg16-mnist.h5")
            model._name = "vgg16"
        elif dataset.lower() == "cifar10":
            model = eval(model_name + "(weights='imagenet', input_shape=(32,32,3), classes=10, include_top=False)")
        elif dataset.lower() == "cifar100":
            model = eval(model_name + "(weights='imagenet', input_shape=(32,32,3), classes=100, include_top=False)")
        elif dataset.lower() == "caltech":
            model = eval(model_name + "(weights='imagenet', input_shape=(224,224,3), classes=102, include_top=False)")
        else:
            model = eval(model_name + "(weights='imagenet')")
    return model
