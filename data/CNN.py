from tensorflow.keras.applications import *
from tensorflow.keras.models import load_model
import gdown
import os

drive = {"vgg16-mnist.h5": 'https://drive.google.com/uc?id=1ziOvcnj973dWlLcG2E7MfqW-RQ18sq5G',
         "vgg16-cifar10.h5": 'https://drive.google.com/uc?id=11CdPEv1gsWGTQqy5AcphzhVyZUY3aENa',
         "vgg16-cifar100.h5": 'https://drive.google.com/uc?id=1-F3iGDWf47MWmdL0p9KKQr26EBBQTtu0'}


def get_model(model_name, model_path, dataset):
    dataset = dataset.lower()
    if model_path:
        model = load_model(model_path)
    else:
        # custom models
        full_model = model_name.lower() + "-" + dataset + ".h5"
        if full_model in drive:
            print("SETUP CUSTOM MODEL")
            if not os.path.exists("data/models/" + full_model):
                url = drive[full_model]
                output = "data/models/" + full_model
                gdown.download(url, output, quiet=False)
            model = load_model("data/models/" + full_model)
            model._name = model_name.lower()
        else:
            print("DEFAULT MODEL WITH IMAGENET WEIGHTS")
            if dataset == "mnist":
                model = eval(model_name + "(weights='imagenet', input_shape=(28,28,1), classes=10, include_top=False)")
            elif dataset == "cifar10":
                model = eval(model_name + "(weights='imagenet', input_shape=(32,32,3), classes=10, include_top=False)")
            elif dataset == "cifar100":
                model = eval(model_name + "(weights='imagenet', input_shape=(32,32,3), classes=100, include_top=False)")
            elif dataset == "caltech":
                model = eval(model_name + "(weights='imagenet', input_shape=(224,224,3), classes=102, include_top=False)")
            else:
                model = eval(model_name + "(weights='imagenet')")
    return model
