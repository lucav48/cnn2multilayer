from sklearn.metrics import auc
from Prediction import *


def average_drop_increase(heatmap, img, model, dataset, ground_class, labels):
    curve = {}
    for th in np.arange(0, 1.05, 0.1):
        top_pixels = {(i, j): heatmap[i, j] for i in range(heatmap.shape[0]) for j in range(heatmap.shape[1])}
        top_pixels = dict(sorted(top_pixels.items(), key=lambda item: item[1], reverse=True))
        n_pixels = int(len(top_pixels) * th)
        top_pixels = {k: top_pixels[k] for k in list(top_pixels)[:n_pixels]}
        mask = np.zeros((img.shape[1:][0], img.shape[1:][1], 3), dtype=int)
        for p in top_pixels:
            mask[p[0], p[1], 0] = 1
            mask[p[0], p[1], 1] = 1
            mask[p[0], p[1], 2] = 1
        l_clean, c_clean = predict_all(dataset, model, img, ground_class, labels)
        l_mask, c_mask = predict_all(dataset, model, img * mask, ground_class, labels)
        curve[th] = (c_clean, c_mask)
    # increase
    inc = 0
    for th, (c1, c2) in curve.items():
        inc += max([0, (c1 - c2)]) / c1
    inc = inc / len(curve) * 100
    # drop
    drop = 0
    for th, (c1, c2) in curve.items():
        drop += abs(c1 < c2)
    drop = drop / len(curve) * 100
    return inc, drop


def deletion_metric(heatmap, img, model, dataset, ground_class, labels):
    curve = {}
    for th in np.arange(0, 1.05, 0.1):
        low_pixels = {(i, j): heatmap[i, j] for i in range(heatmap.shape[0]) for j in range(heatmap.shape[1])}
        low_pixels = dict(sorted(low_pixels.items(), key=lambda item: item[1]))
        n_pixels = int(len(low_pixels) * th)
        low_pixels = {k: low_pixels[k] for k in list(low_pixels)[:n_pixels]}
        mask = np.ones((img.shape[1:][0], img.shape[1:][1], 3), dtype=int)
        for p in low_pixels:
            mask[p[0], p[1], 0] = 0
            mask[p[0], p[1], 1] = 0
            mask[p[0], p[1], 2] = 0
        l, c = predict_all(dataset, model, img * mask, ground_class, labels)
        curve[th] = c
    return curve, auc(list(curve.keys()), list(curve.values()))


def insert_metric(heatmap, img, model, dataset, ground_class, labels):
    curve = {}
    for th in np.arange(0, 1.05, 0.1):
        top_pixels = {(i, j): heatmap[i, j] for i in range(heatmap.shape[0]) for j in range(heatmap.shape[1])}
        top_pixels = dict(sorted(top_pixels.items(), key=lambda item: item[1], reverse=True))
        n_pixels = int(len(top_pixels) * th)
        top_pixels = {k: top_pixels[k] for k in list(top_pixels.keys())[:n_pixels]}
        mask = np.zeros((img.shape[1:][0], img.shape[1:][1], 3), dtype=int)
        for p in top_pixels:
            mask[p[0], p[1], 0] = 1
            mask[p[0], p[1], 1] = 1
            mask[p[0], p[1], 2] = 1
        l, c = predict_all(dataset, model, img * mask, ground_class, labels)
        curve[th] = c
    return curve, auc(list(curve.keys()), list(curve.values()))
