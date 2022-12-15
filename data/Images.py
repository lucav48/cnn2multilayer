from tensorflow.keras.datasets import cifar10, cifar100
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import preprocess_input as vgg16_preprocess_input
from tensorflow.keras.applications.vgg19 import preprocess_input as vgg19_preprocess_input
from tensorflow.keras.applications.resnet import preprocess_input as resnetv1_preprocess_input
from tensorflow.keras.applications.resnet_v2 import preprocess_input as resnetv2_preprocess_input
from tensorflow.keras.applications.vgg16 import decode_predictions as vgg16_decode_predictions
from tensorflow.keras.applications.resnet import decode_predictions as resnet_decode_predictions
from mlxtend.data import loadlocal_mnist
import os
import numpy as np
import gdown
import tarfile


def get_images(dataset, model_name):
    if dataset.lower() == "cifar10":
        num_classes = 10
        (x_train, y_train), _ = cifar10.load_data()
        y_train = y_train.flatten()
    elif dataset.lower() == "cifar100":
        num_classes = 100
        (x_train, y_train), _ = cifar100.load_data()
        y_train = y_train.flatten()
    elif dataset.lower() == "mnist":
        num_classes = 10
        x_train, y_train = loadlocal_mnist(
            images_path='data/mnist/mnist-train-images',
            labels_path='data/mnist/mnist-train-labels')
        x_train = np.array([x.reshape(28, 28, 1).astype('float32') for x in x_train])
        x_train = x_train / 255
    elif dataset.lower() == "caltech":
        num_classes = 101
        if len(os.listdir("data/caltech")) < 2:
            # empty folder, download dataset
            url = 'https://drive.google.com/uc?id=1n28fjRzJ3xpoETicjcs6WxfVa3l8mrwY'
            output = "data/caltech/101_ObjectCategories.tar.gz"
            gdown.download(url, output, quiet=False)
            # extract
            f = tarfile.open("data/caltech/101_ObjectCategories.tar.gz")
            # extracting file
            f.extractall("data/caltech/")
            f.close()

        x_train = []
        y_train = []
        base_path = "data/caltech/101_ObjectCategories"
        i = 0
        for directory in sorted(os.listdir(base_path)):
            n_img = 0
            for img in os.listdir(base_path + "/" + directory):
                img = image.load_img(base_path + "/" + directory + "/" + img, target_size=(224, 224))
                x = image.img_to_array(img)
                x_train.append(x)
                y_train.append(i)
                n_img += 1
                if n_img > 5:
                    break
            i += 1
        num_classes = len(set(y_train))
        x_train, y_train = np.array(x_train), np.array(y_train)
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

    x_train = preprocess(x_train, dataset, model_name)
    return x_train, y_train, num_classes


def preprocess(x, dataset, model_name):
    if "resnet" in model_name.lower() and "v2" in model_name.lower():
        x = resnetv2_preprocess_input(x)
    elif "resnet" in model_name.lower():
        x = resnetv1_preprocess_input(x)
    elif dataset.lower() == "mnist":
        x = x / 255
    else:
        x = eval(model_name + "_preprocess_input" + "(x)")
    return x