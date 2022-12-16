import math

import pandas as pd
import numpy as np
from keras import backend as K


def get_cnn_activations(model, images):
    # define a function to get the activation of all layers
    outputs = []
    for i in range(1, len(model.layers)):
        if model.layers[i].__class__.__name__ == "Conv2D":
            outputs.append(model.layers[i - 1].output)
    active_func = K.function([model.input], [outputs])

    # initialize activations
    all_activations = {}
    for o in outputs:
        all_activations[o.name.split("/")[0]] = np.zeros((1, *tuple([x for x in o.shape[1:]])))

    for img in images:
        activations = active_func(img.reshape(1, *img.shape))
        for layer, j in zip(activations[0][0], range(len(activations[0][0]))):
            all_activations[list(all_activations.keys())[j]] = all_activations[list(all_activations.keys())[j]] + layer

    # take the mean
    for k, v in all_activations.items():
        all_activations[k] = v / len(images)
    return all_activations


def compute_weights_graph(model, images, patched_layers, aggregation):
    data = {}
    layers_name = [x.name for x in model.layers]
    activations = get_cnn_activations(model, images)
    for i in range(1, len(patched_layers)):
        example_source = patched_layers[i - 1][0]  # get info about filters
        if layers_name.index(example_source["layer_name"]) - 1 < 0:
            # actual layer has no predecessor
            continue
        weights_id = layers_name[layers_name.index(example_source["layer_name"]) - 1]
        activation_map = activations[weights_id]
        source_img_shape = example_source["width"], example_source["height"]
        target_img_shape = activation_map.shape[1:3]
        # scostamento da fare
        conv_width = int((example_source['filter_width'] - 1) / 2)
        conv_height = int((example_source['filter_height'] - 1) / 2)
        if conv_width == 0:
            conv_width = 1
            conv_height = 1
        same_shape = source_img_shape == target_img_shape
        for node in patched_layers[i - 1]:
            if same_shape:
                sub_map = activation_map[:, node["x"] - conv_width:node["x"] + conv_width + 1,
                          node["y"] - conv_height:node["y"] + conv_height + 1, :]
            else:
                # max pooling
                width_ratio = source_img_shape[0] / target_img_shape[0]
                height_ratio = source_img_shape[1] / target_img_shape[1]
                source_x_left = int((node["x"] - conv_width) / width_ratio)
                source_x_right = int((node["x"] + conv_width) / width_ratio)
                source_y_left = int((node["y"] - conv_height) / height_ratio)
                source_y_right = int((node["y"] + conv_height) / height_ratio)

                sub_map = activation_map[:, source_x_left:source_x_right,
                          source_y_left:source_y_right, :]
            # features to add to the graph
            if sub_map.size == 0:
                activation = 0
            else:
                if aggregation == "mean":
                    activation = round(np.mean(sub_map), 3)
                elif aggregation == "max":
                    activation = round(np.max(sub_map), 3)
                elif aggregation == "sum":
                    activation = round(np.sum(sub_map), 3)
                elif aggregation == "entropy":
                    filter = np.sum(np.sum(sub_map, axis=2), axis=1)
                    total = np.sum(filter)
                    max_e = 0
                    index_e = 0
                    for i, f in enumerate(filter[0]):
                        try:
                            e = - 1 * (f / total * math.log2(f / total))
                            if e > max_e:
                                max_e = float(e)
                                index_e = int(i)
                        except:
                            continue
                    activation = max_e

            data[node["id"]] = activation
    data = pd.DataFrame.from_dict(data, orient="index").rename(columns={0: "output"})
    return data
