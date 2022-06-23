import tensorflow as tf
from tensorflow.keras.layers import Concatenate, Conv2D, MaxPooling2D, GlobalAveragePooling2D, TimeDistributed, add
from tensorflow.keras.applications import EfficientNetB0, MobileNetV2
from tensorflow.keras import layers

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

def get_efficientnet_model(input, prefix_name):
    input_filter = tf.keras.layers.Conv2D(3, 3, padding='same', name=prefix_name+'_filter')(input)

    base_model = EfficientNetB0(weights=None, pooling='max', include_top=False)(input_filter)

    # base_model._name = prefix_name + base_model._name
    # for layer in base_model.layers:
    #     layer._name = prefix_name + str(layer.name)

    model = base_model(input_filter)
    return model

def get_mobilenet_model(input, prefix_name, fine_tune=False):
    input_filter = tf.keras.layers.Conv2D(3, 3, padding='same', name=prefix_name+'_filter')(input)

    base_model = MobileNetV2(pooling='max', weights='imagenet', include_top=False)
    base_model._name = prefix_name + base_model._name
    
    for layer in base_model.layers:
        layer._name = prefix_name + str(layer.name)
    
    if fine_tune:
        for layer in base_model.layers[60:]:
            if not isinstance(layer, layers.BatchNormalization):
                layer.trainable = True
    else:
        base_model.trainable = True

    model = base_model(input_filter)

    return model
