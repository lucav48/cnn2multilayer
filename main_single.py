import argparse
import os.path

import networkx as nx
import tqdm
import pandas as pd

from data import Images
from data.CNN import get_model
from multilayer.Arc import compute_weights_graph
from multilayer.Graph import create_graph
from data.Images import get_images
from multilayer.Patch import get_patched_layers
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt
import numpy as np

sna_dataset_path = "output/"


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', help='Model name from Keras (VGG16, ResNet50)')
    parser.add_argument('--model_path', help='Model path to load')
    parser.add_argument('--dataset', help='Dataset for the multilayer arcs weights (CIFAR10, CIFAR100, IMAGENET)', required=True)
    parser.add_argument('--image_path', help='Specify path of the image', required=True)
    args = parser.parse_args()

    model = get_model(args.model_name, args.model_path, args.dataset)
    dataset = args.dataset
    model_name = model.name

    image = image.load_img(args.image_path, target_size=(224, 224))
    image = np.expand_dims(image, axis=0)
    image = Images.preprocess(image, dataset, model_name)
    _, _, num_classes = get_images(args.dataset, model_name)

    # create graph or load the one already created
    model_path = sna_dataset_path + dataset + "/" + dataset + "-" + model_name + ".gexf"
    if not os.path.exists(model_path):
        print("CREATING", model_path)
        cnn_graph, patched_layers = create_graph(model)
        nx.write_gexf(cnn_graph, model_path, prettyprint=False)
        # save node attributes as csv
        node_attributes = pd.DataFrame.from_dict(dict(cnn_graph.nodes(data=True)), orient='index').reset_index().rename(
            columns={"index": "id"})
        node_attributes.to_csv(sna_dataset_path + dataset + "/" + dataset + "-" + model_name + "_nodes.csv", index=False)
    else:
        patched_layers = get_patched_layers(model)

    # add weights to the graph
    weights_label = compute_weights_graph(model, image, patched_layers)
    weights_label.to_csv(
        sna_dataset_path + dataset + "/" + dataset + "-" + model_name + "_" +
        args.image_path.split("/")[-1].split(".")[0] + ".csv")

