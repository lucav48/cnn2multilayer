import os

import pandas as pd

from data import Images
from visual.Heatmap import compute_heatmap
from visual.NetworkPath import single_network, compute_paths
from tensorflow.keras.preprocessing import image as keras_image
import numpy as np
from data.Images import preprocess
from Metrics import *
from data.CNN import get_model


def single_run(model_name, dataset, single_path):
    # compute the class network
    command = "python main_single.py --model_name " + model_name + " " + \
              " --dataset " + dataset + " --aggregation entropy" +\
              " --image_path " + single_path
    os.system(command)

    # load the class network
    labels = Images.get_labels(model_name)
    multilayer, nodes_attributes = single_network(single_path, dataset, model_name, labels)

    # load ground truth, image and preprocessed image
    ground_class = list(multilayer.keys())[0]
    image = keras_image.load_img(single_path, target_size=(224, 224))
    image = np.expand_dims(image, axis=0)
    p_img = preprocess(image.copy(), dataset, model_name)

    # compute paths and heatmap
    pixels = compute_paths(multilayer, model_name, ground_class, nodes_attributes, max_per_node=1)
    heatmap, count_pixels = compute_heatmap(image, pixels)

    # metrics
    model = get_model(model_name, model_path="", dataset=dataset)
    inc, drop = average_drop_increase(heatmap, p_img, model, dataset, ground_class, labels)
    curve_insert, auc_insert = insert_metric(heatmap, p_img, model, dataset, ground_class, labels)
    curve_deletion, auc_deletion = deletion_metric(heatmap, p_img, model, dataset, ground_class, labels)
    return (inc, drop), (curve_insert, auc_insert), (curve_deletion, auc_deletion)


dataset = "imagenet"
model_name = "VGG16"
# single_path = "data/imagen/n01443537_5048_goldfish.jpg"

for single_path in os.listdir("data/imagen"):
    avg_inc_drop, insert, deletion = single_run(model_name, dataset, single_path)

    image_name = single_path.split("/")[-1].replace(".jpg", "")
    d = {
         "auc_inc": avg_inc_drop[0],
         "auc_drop": avg_inc_drop[1],
         "curve_insert": insert[0],
         "auc_insert": insert[1],
         "curve_deletion": deletion[0],
         "auc_deletion": deletion[1]
         }
    d = pd.DataFrame.from_dict(d)
    d.to_csv("visual/results/" + image_name + ".csv")
