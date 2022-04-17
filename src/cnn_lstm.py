import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Concatenate, Conv2D, MaxPooling2D, GlobalAveragePooling2D, TimeDistributed

MAX_SEQ_LENGTH = 16
NUM_FEATURES = 3000
NUMBER_OF_CLASSES = 3
HAND_WIDTH, HAND_HEIGHT = 80, 80
FACE_WIDTH, FACE_HEIGHT = 80, 80

def get_sequence_model():
    cnn_model = get_cnn_model()

    frame_features_input = [keras.Input((16, HAND_WIDTH, HAND_HEIGHT, 3), name="input"+str(c)) for c in range(3)]

    # Refer to the following tutorial to understand the significance of using `mask`:
    # https://keras.io/api/layers/recurrent_layers/gru/
    x = TimeDistributed(cnn_model)(frame_features_input)
    x = keras.layers.GRU(16, return_sequences=True)(x)
    x = keras.layers.GRU(8)(x)
    x = keras.layers.Dropout(0.4)(x)
    x = keras.layers.Dense(8, activation="relu")(x)
    output = keras.layers.Dense(NUMBER_OF_CLASSES, activation="softmax")(x)

    rnn_model = keras.Model(frame_features_input, output)

    rnn_model.compile(
        loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["accuracy"]
    )

    tf.keras.utils.plot_model(rnn_model, "model.png", show_shapes=True)
    return rnn_model


def get_base_sequence(input):
    x = Conv2D(filters=64, kernel_size=3, padding="same",activation="relu")(input)
    x = MaxPooling2D(pool_size=2)(x)
    x = Conv2D(filters=32, kernel_size=3, padding="same",activation="relu")(x)
    x = MaxPooling2D(pool_size=2)(x)
    x = Conv2D(filters=16, kernel_size=3, padding="same",activation="relu")(x)
    x = MaxPooling2D(pool_size=2)(x)
    output = GlobalAveragePooling2D()(x)
    return output

def get_cnn_model():
    hand1_input = tf.keras.layers.Input(shape=(HAND_WIDTH, HAND_HEIGHT, 3), name='hand1_input')
    hand2_input = tf.keras.layers.Input(shape=(HAND_WIDTH, HAND_HEIGHT, 3), name='hand2_input')
    face_input = tf.keras.layers.Input(shape=(FACE_WIDTH, FACE_HEIGHT, 3), name='face_input')

    hand1_seq = get_base_sequence(hand1_input)
    hand2_seq = get_base_sequence(hand2_input)
    face_seq = get_base_sequence(face_input)

    final_output = Concatenate()([hand1_seq, hand2_seq, face_seq])

    model = Model(inputs=[hand1_input, hand2_input, face_input], outputs=final_output)

    tf.keras.utils.plot_model(model, "model.png", show_shapes=True)

    return model

get_sequence_model()