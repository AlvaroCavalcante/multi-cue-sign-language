import tensorflow as tf
from tensorflow.keras.layers import Concatenate, Conv2D, MaxPooling2D, GlobalAveragePooling2D, TimeDistributed, add
from tensorflow.keras.applications import EfficientNetB0

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
    x = Conv2D(filters=82, kernel_size=3,
               padding="same", activation="relu")(input)
    x = MaxPooling2D(pool_size=2)(x)
    x = Conv2D(filters=42, kernel_size=3, padding="same", activation="relu")(x)
    x = MaxPooling2D(pool_size=2)(x)
    x = Conv2D(filters=16, kernel_size=3, padding="same", activation="relu")(x)
    x = MaxPooling2D(pool_size=2)(x)

    output = GlobalAveragePooling2D()(x)
    return output

def get_efficient_net_model(input, width, height):
    hand_filter = tf.keras.layers.Conv2D(3, 3, padding='same', name='hand_filter')(input)

    hand_model = EfficientNetB0(input_shape=(width, height, 3),include_top=False)(hand_filter)
    output = GlobalAveragePooling2D()(hand_model)
    return output
