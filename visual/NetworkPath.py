import networkx as nx
import pandas as pd
from tqdm import tqdm
import numpy as np


def single_network(single_path, dataset, model_name, labels, base_graph=None):
    graphs = {}
    path = "output/" + dataset + "/"
    nodes_attributes = pd.read_csv(path + dataset + "-" + model_name.lower() + "_nodes.csv", index_col="id")
    nodes_attributes.index = nodes_attributes.index.astype(str)
    nodes_attributes = nodes_attributes.to_dict(orient="index")
    if not base_graph:
        base_graph = nx.read_gexf(path + dataset + "-" + model_name.lower() + ".gexf")
    filename = single_path.split("/")[-1].replace(".jpg", ".csv")
    csv_file = path + dataset + "-" + model_name.lower() + "_" + filename
    df = pd.read_csv(csv_file, index_col=0)

    graph_id = int(labels[single_path.split("/")[-1].split("_")[0]])
    class_graph = base_graph.copy()
    # update weights
    for n1 in class_graph.nodes:
        out_n = list(class_graph.neighbors(n1))
        if len(out_n) > 0:
            w = df.loc[int(n1)]["output"]
            for n2 in out_n:
                class_graph[n1][n2]["weight"] = w
    graphs[graph_id] = class_graph.copy()
    nx.set_node_attributes(graphs[graph_id], nodes_attributes)
    return graphs, nodes_attributes


def compute_paths(graphs, model_name, image_class, nodes_attributes, max_per_node=4, th_paths=0.75):
    one_key = list(graphs.keys())[0]
    first_layer = graphs[one_key].nodes[list(graphs[one_key].nodes)[0]]["layer_name"]
    last_layer = graphs[one_key].nodes[list(graphs[one_key].nodes)[-1]]["layer_name"]

    nodes_first_layer = [x[0] for x in graphs[one_key].nodes(data=True) if x[1]["layer_name"] == first_layer]
    nodes_last_layer = [x[0] for x in graphs[one_key].nodes(data=True) if x[1]["layer_name"] == last_layer]

    print("FIRST:", first_layer, "(", len(nodes_first_layer), ") LAST:", last_layer, "(", len(nodes_last_layer), ")")

    # ---------------- SECONDO GREEDY ------------------
    w_path = {}

    for n in tqdm(nodes_last_layer[::10]):
        explore = {x[0]: x[2]["weight"] for x in graphs[image_class].in_edges(n, data=True)}
        explore = dict(sorted(explore.items(), key=lambda item: item[1], reverse=True))
        explore = {k: explore[k] for k in list(explore.keys())[:max_per_node]}
        while len(explore) > 0:
            node, weight = list(explore.keys())[0], list(explore.values())[0]
            del explore[node]
            nodes_explore = {x[0]: x[2]["weight"] for x in graphs[image_class].in_edges(node, data=True)}
            if len(nodes_explore) > 0:
                nodes_explore = dict(sorted(nodes_explore.items(), key=lambda item: item[1], reverse=True))
                nodes_explore = {k: nodes_explore[k] + weight for k in list(nodes_explore.keys())[:max_per_node]}
                explore.update(nodes_explore)
            else:
                if node in w_path:
                    w_path[node] += weight
                else:
                    w_path[node] = weight

    # extract pixels
    pixels = {}
    th_weights = np.quantile(list(w_path.values()), th_paths)
    for n, w in w_path.items():
        if w > th_weights:
            if "resnet" in model_name:
                # resnet ha subito un conv ed un max pooling. devo convertire le coordinate post
                # max pooling in coordinate dell'immagine in input
                x, y = (nodes_attributes[n]["x"] - 1) * 2, (nodes_attributes[n]["y"] - 1) * 2
            else:
                x, y = nodes_attributes[n]["x"], nodes_attributes[n]["y"]
            pixels[(x, y)] = w

    print("N PATHS:", len(pixels))
    return pixels
