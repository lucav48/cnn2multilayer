from tensorflow.keras.datasets import cifar10, cifar100
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import preprocess_input as vgg16_preprocess_input
from tensorflow.keras.applications.resnet import preprocess_input as resnetv1_preprocess_input
from tensorflow.keras.applications.resnet_v2 import preprocess_input as resnetv2_preprocess_input
from tensorflow.keras.applications.vgg16 import decode_predictions as vgg16_decode_predictions
from tensorflow.keras.applications.resnet import decode_predictions as resnet_decode_predictions
import os
import numpy as np


def get_images(dataset, model_name):
    if dataset.lower() == "cifar10":
        num_classes = 10
        (x_train, y_train), _ = cifar10.load_data()
    elif dataset.lower() == "cifar100":
        num_classes = 100
        (x_train, y_train), _ = cifar100.load_data()
    elif dataset.lower() == "imagenet":
        if "resnet" in model_name.lower():
            preprocess_function = "resnet_decode_predictions" + "(np.expand_dims(np.arange(1000), 0), top=1000)"
        else:
            preprocess_function = model_name + "_decode_predictions" + "(np.expand_dims(np.arange(1000), 0), top=1000)"

        labels = eval(preprocess_function)
        labels = {k: v for k, _, v in labels[0]}
        x_train = []
        y_train = []
        for img_name in os.listdir("data/imagen"):
            label = img_name.split("_")[0]
            if ".jpg" in img_name and label in labels:
                img = image.load_img("data/imagen/" + img_name, target_size=(224, 224))
                x = image.img_to_array(img)

                x_train.append(x)
                y_train.append(labels[label])
        num_classes = len(set(y_train))
        x_train, y_train = np.array(x_train), np.array(y_train)

    if "resnet" in model_name.lower() and "v2" in model_name.lower():
        x_train = resnetv2_preprocess_input(x_train)
    elif "resnet" in model_name.lower():
        x_train = resnetv1_preprocess_input(x_train)
    else:
        x_train = eval(model_name + "_preprocess_input" + "(x_train)")
    return x_train, y_train, num_classes