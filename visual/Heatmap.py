import numpy as np
from skimage.segmentation import flood_fill
from matplotlib import cm


def compute_heatmap(img, pixels):
    # create heatmap
    count_pixels = {}
    heatmap = np.zeros((img[0].shape[0], img[0].shape[1]), dtype=int)
    for x, y in pixels:
        for i in range(-3, 4, 1):
            for j in range(-3, 4, 1):
                x_i = x + i
                y_j = y + j
                try:
                    if (x_i, y_j) in count_pixels:
                        count_pixels[(x_i, y_j)] += 1
                        heatmap[x_i, y_j] += 1
                        heatmap[x_i, y_j] += 1
                        heatmap[x_i, y_j] += 1
                    else:
                        count_pixels[(x_i, y_j)] = 1
                        heatmap[x_i, y_j] = 1
                        heatmap[x_i, y_j] = 1
                        heatmap[x_i, y_j] = 1
                except:
                    continue
    for p, v in count_pixels.items():
        heatmap = flood_fill(heatmap, seed_point=p, new_value=v, connectivity=1)
    heatmap = heatmap / np.max(heatmap)
    return heatmap, count_pixels


def plot_heatmap(img, normalized_heatmap):
    alpha = 0.4
    heatmap = normalized_heatmap.copy()
    # Rescale heatmap to a range 0-255
    heatmap = np.uint8(255 * heatmap)

    # Use jet colormap to colorize heatmap
    jet = cm.get_cmap("jet")

    # Use RGB values of the colormap
    jet_colors = jet(np.arange(256))[:, :3]
    jet_heatmap = jet_colors[heatmap]

    # Create an image with RGB colorized heatmap
    jet_heatmap = array_to_img(jet_heatmap)
    jet_heatmap = jet_heatmap.resize((img.shape[1], img.shape[2]))
    jet_heatmap = img_to_array(jet_heatmap)

    # Superimpose the heatmap on original image
    superimposed_img = jet_heatmap * alpha + img[0]
    superimposed_img = array_to_img(superimposed_img)

    # Display Grad CAM
    fig = plt.figure(figsize=(8, 8))
    plt.imshow(superimposed_img)
