# CNN2Multilayer

Starting from a CNN given in input, this package creates a multilayer network where each node represents a portion of the image and the weights are the convolution results on that portion.

## Requirements
Required libraries: pandas, networkx, tensorflow.

## Usage
main.py is the script to generate a multilayer network starting from a CNN model. It needs 4 parameters:
- model_name: model name from Keras from https://keras.io/api/applications/ (e.g. VGG16, ResNet50). Required if model_path not required. It downloads the pretrained CNN model from keras.
- model_path: model path containing the CNN to load. Actually .h5 file support. Required if model_name not provided.
- dataset: dataset name for the multilayer arcs weights (supporting CIFAR10, CIFAR100, IMAGENET)
- images_range: if you want to study a subset of images (e.g. get the feature maps of few images and put them as weights of the multilayer network), insert here the range of images to pick (e.g. 10-11 for the 10th image, 10-20 for the 10th image to the 19th image of the selected dataset)

**Example run**:
```
python3 main.py --model_name VGG16 --dataset imagenet --images_range 1-2
```

**Result**:
Run main.py, download the VGG16 model from keras api and download the imagenet dataset. Create the multilayer network of VGG16 (stored in a .gexf file), compute the feature maps of the image 1 of the imagenet dataset and add it as weights of the previous multilayer network.

## Compression
Open CNNCompression notebook, setup the folder path, and run all the cells to get the layer removed from the CNN model for each threshold.