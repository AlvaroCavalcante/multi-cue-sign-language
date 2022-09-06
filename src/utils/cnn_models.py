import tensorflow as tf
from tensorflow.keras.layers import Concatenate, Conv2D, MaxPooling2D, GlobalAveragePooling2D, TimeDistributed, add
from tensorflow.keras.applications import EfficientNetB0, MobileNetV2, EfficientNetV2S, Xception
from tensorflow.keras import layers
from utils import utils


def get_toy_rnn(input):
    x = Conv2D(32, 3, activation="relu")(input)
    x = Conv2D(64, 3, activation="relu")(x)
    block_1_output = MaxPooling2D(3)(x)

    x = Conv2D(64, 3, activation="relu", padding="same")(block_1_output)
    x = Conv2D(64, 3, activation="relu", padding="same")(x)
    block_2_output = add([x, block_1_output])

    x = Conv2D(64, 3, activation="relu", padding="same")(block_2_output)
    x = Conv2D(64, 3, activation="relu", padding="same")(x)
    block_3_output = add([x, block_2_output])

    x = Conv2D(64, 3, activation="relu")(block_3_output)
    output = GlobalAveragePooling2D()(x)

    return output


def get_custom_cnn(input):
    x = Conv2D(filters=120, kernel_size=3,
               padding="same", activation="relu")(input)
    x = MaxPooling2D(pool_size=2)(x)
    x = Conv2D(filters=64, kernel_size=3, padding="same", activation="relu")(x)
    x = Conv2D(filters=32, kernel_size=3, padding="same", activation="relu")(x)
    x = MaxPooling2D(pool_size=2)(x)
    x = Conv2D(filters=16, kernel_size=3, padding="same", activation="relu")(x)
    x = MaxPooling2D(pool_size=2)(x)
    output = GlobalAveragePooling2D()(x)
    return output


def get_efficientnet_model(input, prefix_name, fine_tune=False):
    base_model = EfficientNetB0(
        weights='imagenet', pooling='avg', include_top=False)

    base_model._name = prefix_name + base_model._name

    for layer_n, layer in enumerate(base_model.layers):
        # Each block needs to be all turned on or off.
        layer._name = prefix_name + str(layer.name)
        if fine_tune:
            # 75 block 3 # 119 block 4 # 162 block 5 # 221 block 6
            if isinstance(layer, layers.BatchNormalization) or layer_n < 75:
                base_model.layers[layer_n].trainable = False
        else:
            layer.trainable = False

    utils.get_param_count(base_model)

    model = base_model(input)
    return model


def get_xception_model(input, prefix_name, fine_tune=False):
    input_filter = tf.keras.layers.Conv2D(
        3, 3, padding='same', name=prefix_name+'_filter')(input)

    base_model = Xception(
        weights='imagenet', pooling='avg', include_top=False)

    base_model._name = prefix_name + base_model._name

    for layer_n, layer in enumerate(base_model.layers):
        # Each block needs to be all turned on or off.
        layer._name = prefix_name + str(layer.name)
        if False:
            # 75 block 3 # 119 block 4 # 162 block 5 # 221 block 6
            if isinstance(layer, layers.BatchNormalization) or layer_n < 119:
                base_model.layers[layer_n].trainable = False
        else:
            layer.trainable = False

    utils.get_param_count(base_model)

    model = base_model(input_filter)
    return model


def get_efficientnet_v2_model(input, prefix_name, fine_tune=False):
    input_filter = tf.keras.layers.Conv2D(
        3, 3, padding='same', name=prefix_name+'_filter')(input)

    base_model = EfficientNetV2S(
        weights='imagenet', pooling='avg', include_top=False)

    base_model._name = prefix_name + base_model._name

    for layer_n, layer in enumerate(base_model.layers):
        # Each block needs to be all turned on or off.
        layer._name = prefix_name + str(layer.name)
        if fine_tune:
            # 267 # block 5
            if isinstance(layer, layers.BatchNormalization) or layer_n < 322:
                base_model.layers[layer_n].trainable = False
        else:
            layer.trainable = False

    utils.get_param_count(base_model)

    model = base_model(input_filter)
    return model


def get_mobilenet_model(input, prefix_name, fine_tune=False):
    input_filter = tf.keras.layers.Conv2D(
        3, 3, padding='same', name=prefix_name+'_filter')(input)

    base_model = MobileNetV2(
        pooling='avg', weights='imagenet', include_top=False)
    base_model._name = prefix_name + base_model._name

    for layer_n, layer in enumerate(base_model.layers):
        layer._name = prefix_name + str(layer.name)

        if fine_tune:
            # 54 Layer 5 # 63 Layer 6 # 107, #134 # 142
            if isinstance(layer, layers.BatchNormalization) or layer_n < 143:
                layer.trainable = False
        else:
            layer.trainable = False

    utils.get_param_count(base_model)
    model = base_model(input_filter)

    return model
