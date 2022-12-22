import os
import pandas as pd
from data import Images
from visual.GradCAM import make_gradcam_heatmap
from visual.Heatmap import compute_heatmap
from visual.NetworkPath import single_network, compute_paths
from tensorflow.keras.preprocessing import image as keras_image
import numpy as np
from data.Images import preprocess
from Metrics import *
from data.CNN import get_model
from tqdm import tqdm
import networkx as nx


def single_run(model_name, dataset, single_path, base_graph, max_per_node=1):
    # compute the class network
    existing_path = dataset + "-" + model_name.lower() + "_" + single_path.split("/")[-1].replace(".jpg", ".csv")
    if existing_path not in os.listdir("output/" + dataset + "/"):
        command = "python main_single.py --model_name " + model_name + " " + \
                  " --dataset " + dataset + " --aggregation entropy" + \
                  " --image_path " + single_path
        os.system(command)

    # load the class network
    labels = Images.get_labels(model_name)
    multilayer, nodes_attributes = single_network(single_path, dataset, model_name, labels,
                                                  base_graph=base_graph)

    # load ground truth, image and preprocessed image
    ground_class = list(multilayer.keys())[0]
    image = keras_image.load_img(single_path, target_size=(224, 224))
    image = np.expand_dims(image, axis=0)
    p_img = preprocess(image.copy(), dataset, model_name)

    # compute paths and heatmap
    pixels = compute_paths(multilayer, model_name, ground_class, nodes_attributes, max_per_node=max_per_node)
    heatmap, count_pixels = compute_heatmap(image, pixels)

    # our method metrics
    model = get_model(model_name, model_path="", dataset=dataset)
    inc, drop = average_drop_increase(heatmap, p_img, model, dataset, ground_class, labels)
    curve_insert, auc_insert = insert_metric(heatmap, p_img, model, dataset, ground_class, labels)
    curve_deletion, auc_deletion = deletion_metric(heatmap, p_img, model, dataset, ground_class, labels)

    # competitor
    last_node = list(multilayer[ground_class].nodes)[-1]
    last_layer = multilayer[ground_class].nodes[last_node]["layer_name"]
    grad = make_gradcam_heatmap(image, model, last_layer, pred_index=ground_class)
    resized_grad = grad.copy()
    resized_grad.resize((image[0].shape[0], image[0].shape[1]))
    grad_inc, grad_drop = average_drop_increase(resized_grad, p_img, model, dataset, ground_class, labels)
    grad_curve_insert, grad_auc_insert = insert_metric(resized_grad, p_img, model, dataset, ground_class, labels)
    grad_curve_deletion, grad_auc_deletion = deletion_metric(resized_grad, p_img, model, dataset, ground_class, labels)

    return [(inc, drop), (curve_insert, auc_insert), (curve_deletion, auc_deletion)], \
           [(grad_inc, grad_drop), (grad_curve_insert, grad_auc_insert), (grad_curve_deletion, grad_auc_deletion)]


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
dataset = "imagenet"
model_name = "VGG16"
max_per_node = 3
# single_path = "data/imagen/n01443537_5048_goldfish.jpg"
base_path = "data/imagen/"
base_graph = nx.read_gexf("output/" + dataset + "/" + dataset + "-" + model_name.lower() + ".gexf")

c = [x.split("_")[0] for x in os.listdir("data/imagen")]
intersection_images = set(Images.get_labels(model_name).keys()).intersection(set(c))

count = 0
for single_path in tqdm(os.listdir("data/imagen")):
    try:
        c_path = single_path.split("_")[0]
        if c_path not in intersection_images:
            continue
        mine, competitor = single_run(model_name, dataset, base_path + single_path, base_graph=base_graph,
                                      max_per_node=max_per_node)
        avg_inc_drop, insert, deletion = mine
        grad_avg, grad_insert, grad_deletion = competitor
        image_name = single_path.split("/")[-1].replace(".jpg", "")
        d = {
            "auc_inc": avg_inc_drop[0],
            "auc_drop": avg_inc_drop[1],
            "curve_insert": insert[0],
            "auc_insert": insert[1],
            "curve_deletion": deletion[0],
            "auc_deletion": deletion[1],

            "grad_auc_inc": grad_avg[0],
            "grad_auc_drop": grad_avg[1],
            "grad_curve_insert": grad_insert[0],
            "grad_auc_insert": grad_insert[1],
            "grad_curve_deletion": grad_deletion[0],
            "grad_auc_deletion": grad_deletion[1],
        }
        d = pd.DataFrame.from_dict(d)
        d.to_csv("visual/results/" + image_name + "_" + str(max_per_node) + ".csv")
        count += 1
        if count >= 30:
            break
    except Exception as e:
        print(single_path, e)
