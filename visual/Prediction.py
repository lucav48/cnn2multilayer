from tensorflow.keras.models import load_model
from tensorflow.keras.applications.vgg16 import preprocess_input as vgg16_preprocess_input
from tensorflow.keras.applications.vgg19 import preprocess_input as vgg19_preprocess_input
from tensorflow.keras.applications.resnet import preprocess_input as resnetv1_preprocess_input
from tensorflow.keras.applications.resnet_v2 import preprocess_input as resnetv2_preprocess_input
from tensorflow.keras.applications.vgg16 import decode_predictions as vgg16_decode_predictions
from tensorflow.keras.applications.resnet import decode_predictions as resnet_decode_predictions
import numpy as np


def model_prediction(dataset, model_name, img, labels):
    if dataset == "imagenet":
        keras_model = eval(model_name + "(weights='imagenet')")
    else:
        keras_model = load_model("/content/cnn2multilayer/data/models/" + model_name.lower() + "-" + dataset + ".h5")

    if dataset.lower() == "imagenet":
        command = "decode_predictions(keras_model.predict(img))[0]"
        predicted_image_class = eval(model_name.lower() + "_" + command)
        label_image = labels[predicted_image_class[0][0]]
        confidence_image = predicted_image_class[0][2]
    else:
        predicted_image_class = np.argmax(keras_model.predict(img))
    return label_image, confidence_image


def predict(dataset, model, img, labels):
    if dataset.lower() == "imagenet":
        command = "decode_predictions(model.predict(img))[0]"
        predicted_image_class = eval(model.name + "_" + command)
        label_image = labels[predicted_image_class[0][0]]
        confidence_image = predicted_image_class[0][2]
    else:
        predicted_image_class = np.argmax(model.predict(img))
    return label_image, confidence_image


def predict_all(dataset, model, img, target, labels):
    if dataset.lower() == "imagenet":
        command = "decode_predictions(model.predict(img, verbose=0))[0]"
        predicted_image_class = eval(model.name + "_" + command)
        predictions = {}
        for k, v, c in predicted_image_class:
            predictions[labels[k]] = c
        if target in predictions:
            label_image = target
            confidence_image = predictions[target]
        else:
            label_image = -1
            confidence_image = 0
    else:
        predicted_image_class = model.predict(img, verbose=0)[target]
    return label_image, confidence_image
