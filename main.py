import argparse
import os.path

import networkx as nx
import tqdm
import pandas as pd
from data.CNN import get_model
from multilayer.Arc import compute_weights_graph
from multilayer.Graph import create_graph
from data.Images import get_images
from multilayer.Patch import get_patched_layers

sna_dataset_path = "output/"


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', help='Model name from Keras (VGG16, ResNet50)')
    parser.add_argument('--model_path', help='Model path to load')
    parser.add_argument('--dataset', help='Dataset for the multilayer arcs weights (CIFAR10, CIFAR100, IMAGENET)', required=True)
    parser.add_argument('--images_range', help='Range of images to select (e.g. 10-11 for the 10th image, 10-20 for '
                                               'the 10th image to the 19th image', required=True)
    args = parser.parse_args()

    model = get_model(args.model_name, args.model_path)
    dataset = args.dataset
    model_name = model.name

    x, y, num_classes = get_images(args.dataset, model_name)
    image_range = args.images_range.split("-")

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

    for label in tqdm.tqdm(range(num_classes)):
        images = x[y == label]
        images = images[int(image_range[0]):int(image_range[1])]
        if len(images) > 0:
            # add weights to the graph
            weights_label = compute_weights_graph(model, images, patched_layers)
            weights_label.to_csv(
                sna_dataset_path + dataset + "/" + dataset + "-" + model_name + "_" + str(label) + ".csv")

