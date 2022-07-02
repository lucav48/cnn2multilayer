# creazione del grafo a partire dalle patch
from multilayer.Patch import get_patched_layers
import networkx as nx
import pandas as pd


def create_graph(model):
    patched_layers = get_patched_layers(model)
    # create graph
    cnn_graph = nx.DiGraph()
    for i in range(1, len(patched_layers)):
        print("LAYER: ", i)
        source = pd.DataFrame(patched_layers[i - 1])
        target = pd.DataFrame(patched_layers[i])
        source_name = source.iloc[0]['layer_name']
        target_name = target.iloc[0]['layer_name']

        source_width = patched_layers[i - 1][0]["width"]
        target_width = patched_layers[i][0]["width"]
        source_height = patched_layers[i - 1][0]["height"]
        target_height = patched_layers[i][0]["height"]
        width_ratio = int(source_width / target_width)
        height_ratio = int(source_height / target_height)

        # add nodes
        if i == 1:
            for index, node in source.iterrows():
                cnn_graph.add_node(node['id'], **{'x': node['x'], 'y': node['y'],
                                                  'layer_name': source_name})

        for index, node in target.iterrows():
            cnn_graph.add_node(node['id'], **{'x': node['x'], 'y': node['y'],
                                              'layer_name': target_name})
        print("ADDED NODES")
        filters_width = source.iloc[0]['filter_width']
        filters_height = source.iloc[0]['filter_height']

        conv_width_start = int((filters_width - 1) / 2)
        conv_height_start = int((filters_height - 1) / 2)

        if conv_width_start == 0:
            conv_width_start = 1
            conv_height_start = 1

        for index, t in target.iterrows():
            if width_ratio == 1:
                # immagini di input e output uguali, semplice convoluzione
                conn = source.loc[((source['x'] - conv_width_start) <= t["x"]) &
                                  (t["x"] <= (source['x'] + conv_width_start)) &
                                  ((source['y'] - conv_height_start) <= t["y"]) &
                                  (t["y"] <= (source['y'] + conv_height_start))]
            else:
                # immagini di input e output di diversa dimensione, max pooling
                # prendi le dimensioni originali nelle immagini e disegna un quadrato grande come filters*pool_size
                dim_square = width_ratio * filters_width, height_ratio * filters_height
                source_x_left = (t["x"] - conv_width_start) * width_ratio
                source_x_right = source_x_left + dim_square[0] - 1
                source_y_left = (t["y"] - conv_height_start) * height_ratio
                source_y_right = source_y_left + dim_square[1] - 1

                conn = source.loc[(source['x'] <= source_x_right) &
                                  (source_x_left <= source['x']) &
                                  (source['y'] <= source_y_right) &
                                  (source_y_left <= source['y'])]
            edges = zip(conn['id'].to_list(), [t["id"]] * len(conn))
            cnn_graph.add_edges_from(edges)
    print("ADDED SEQUENTIAL EDGES")
    if "resnet" in model.name:
        cnn_graph = resnet_layers(model, cnn_graph)
    return cnn_graph, patched_layers


def resnet_layers(model, cnn_graph):
    # search adding layers and conv involved
    adds = {}
    for i, layer in enumerate(model.layers):
        kind = layer.__class__.__name__
        # find add layer
        if "add" in kind.lower() and "padding" not in kind.lower():
            # layers before add
            connected = [x.name for x in layer._inbound_nodes[0].inbound_layers]
            convs = []
            last_conv = None
            for x in connected:
                if "conv" in x and "bn" not in x and "add" not in x and "out" not in x:
                    convs.append(x)
                else:
                    for search in model.layers:
                        if "conv" in search.name and "bn" not in search.name and "add" not in search.name and "out" not in search.name:
                            last_conv = search.name
                        elif x == search.name:
                            convs.append(last_conv)
            # layer input to add
            next_conv = None
            for l in model.layers[i:]:
                if "conv" in l.name and "bn" not in l.name and "add" not in l.name and "out" not in l.name and "preact" not in l.name:
                    next_conv = l.name
                    break
            if next_conv:
                adds[layer.name] = {"input": next_conv, "output": convs}

    c = 0
    for add_layer, io in adds.items():
        nodes_input = [x for x in cnn_graph.nodes(data=True) if x[1]["layer_name"] == io["input"]]
        nodes_output = [x for x in cnn_graph.nodes(data=True) if x[1]["layer_name"] == io["output"][0]]
        for out in nodes_output:
            for inp in nodes_input:
                if out[1]["x"] == inp[1]["x"] and out[1]["y"] == inp[1]["y"]:
                    c += 1
                    cnn_graph.add_edge(out[0], inp[0])
    print("ADDED", c, "SKIP-EDGES")
    return cnn_graph
