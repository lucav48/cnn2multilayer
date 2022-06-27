from tensorflow.keras.layers import Conv2D


def get_layer_dimensions(layer):
    layer_name = layer.__class__.__name__
    if layer_name == 'InputLayer':
        width = layer.output_shape[0][1]
        height = layer.output_shape[0][2]
    elif layer_name == 'Conv2D':
        width = layer.output_shape[1]
        height = layer.output_shape[2]
    elif layer_name == 'MaxPooling2D':
        width = layer.output_shape[1]
        height = layer.output_shape[2]
    elif layer_name == 'Flatten':
        pass
    elif layer_name == 'Dense':
        pass
    return width, height


# ---

def get_patched_layers(model):
    id = 1
    patched_layers = []
    #
    layers = [x for x in model.layers if "Conv" in x.__class__.__name__] + [Conv2D(1, kernel_size=(2, 2))]
    for i in range(1, len(layers)):
        source = layers[i - 1]
        target = layers[i]

        source_type = source.__class__.__name__
        source_name = source.name
        width_source, height_source = get_layer_dimensions(source)
        w_filters_target, h_filters_target = get_layer_filters(target)
        pt, id = create_patches(source_name, source_type, width_source, height_source,
                                w_filters_target, h_filters_target, id)
        patched_layers.append(pt)
    return patched_layers


# -------------

def get_layer_filters(layer):
    layer_name = layer.__class__.__name__
    try:
        if layer_name == 'Conv2D':
            width, height = layer.__dict__["kernel_size"]
            # width  = layer.kernel.shape[0]
            # height = layer.kernel.shape[1]
        elif layer_name == 'MaxPooling2D':
            width = layer.pool_size[0]
            height = layer.pool_size[1]
        else:
            width = None
            height = None
    except Exception as e:
        print(e)
        print(layer.__dict__)
        width, height = 0, 0
    return width, height


# -------------

def create_patches(layer_name, layer_id, width_source, height_source, w_filters_target, h_filters_target, id):
    conv_width_start = int((w_filters_target - 1) / 2)
    conv_height_start = int((h_filters_target - 1) / 2)
    conv_width_end = width_source - conv_width_start
    conv_height_end = height_source - conv_height_start
    patches = []
    for i in range(conv_width_start, conv_width_end):
        for j in range(conv_height_start, conv_height_end):
            patch = {'id': id, 'x': i, 'y': j, 'filter_width': w_filters_target, 'filter_height': h_filters_target,
                     'layer_name': layer_name, 'width': width_source, 'height': height_source}
            id = id + 1
            patches.append(patch)
    if len(patches) == 0:
        # too small window
        patches.append({'id': id, 'x': conv_width_start, 'y': conv_height_start, 'filter_width': w_filters_target,
                        'filter_height': h_filters_target,
                        'layer_name': layer_name, 'width': width_source,
                        'height': height_source})
        id = id + 1
    return patches, id
