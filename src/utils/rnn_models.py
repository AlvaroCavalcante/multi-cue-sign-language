import tensorflow as tf
from tensorflow.keras.layers import Layer
from tensorflow.keras import backend as K
from tensorflow import keras
from tensorflow.keras.layers import Bidirectional, LSTM, GRU, Dropout, Dense
from tensorflow.keras.layers import Concatenate, TimeDistributed


FACE_WIDTH, FACE_HEIGHT = 100, 100
MAX_SEQ_LENGTH = 16
NUMBER_OF_CLASSES = 226
HAND_WIDTH, HAND_HEIGHT = 100, 200


class Attention(Layer):

    def __init__(self, return_sequences=True):
        self.return_sequences = return_sequences
        super(Attention, self).__init__()

    def build(self, input_shape):

        self.W = self.add_weight(name="att_weight", shape=(input_shape[-1], 1),
                                 initializer="normal")
        self.b = self.add_weight(name="att_bias", shape=(input_shape[1], 1),
                                 initializer="zeros")

        super(Attention, self).build(input_shape)

    def call(self, x):
        # gerar os pesos das sequência, sendo 1 peso por time-step (frame)
        e = K.tanh(K.dot(x, self.W)+self.b)
        a = K.softmax(e, axis=1)  # coloca os pesos entre 0 e 1
        output = x*a

        if self.return_sequences:
            return output

        return K.sum(output, axis=1)


def get_hands_rnn_model(cnn_model, learning_rate, optimizer='adam'):
    features_input = [keras.Input(
        (16, HAND_WIDTH, HAND_HEIGHT, 3), name="hand_"+str(c)) for c in range(1)]

    x = TimeDistributed(cnn_model)(features_input)
    x = Bidirectional(GRU(288, return_sequences=True))(x)
    x = Dropout(0.25)(x)
    x = Attention(return_sequences=False)(x)

    rnn_model = get_model(learning_rate, optimizer, features_input, x)

    return rnn_model


def get_face_rnn_model(cnn_model, learning_rate, optimizer='adam'):
    features_input = [keras.Input(
        (16, FACE_WIDTH, FACE_HEIGHT, 3), name="face_"+str(c)) for c in range(1)]

    x = TimeDistributed(cnn_model)(features_input)
    x = GRU(384, return_sequences=True)(x)
    x = Dropout(0.05)(x)
    x = GRU(320, return_sequences=True)(x)
    x = Dropout(0.05)(x)
    x = GRU(192, return_sequences=True)(x)
    x = Attention(return_sequences=False)(x)

    rnn_model = get_model(learning_rate, optimizer, features_input, x)

    return rnn_model


def get_triangle_rnn_model(learning_rate, optimizer='adam'):
    tri_input = [keras.Input((16, 13), name='triangle_data')]

    x = GRU(320, return_sequences=True)(tri_input)
    x = Dropout(0.1)(x)
    x = GRU(192, return_sequences=True)(x)
    x = Dropout(0.1)(x)
    x = Attention(return_sequences=False)(x)

    rnn_model = get_model(learning_rate, optimizer, tri_input, x)

    return rnn_model


def get_triangle_figure_rnn_model(cnn_model, learning_rate, optimizer='adam'):
    features_input = [keras.Input(
        (16, 128, 128, 3), name="triangle_fig_"+str(c)) for c in range(1)]

    x = TimeDistributed(cnn_model)(features_input)
    x = Bidirectional(GRU(288, return_sequences=True))(x)
    x = Dropout(0.25)(x)
    x = Bidirectional(GRU(416, return_sequences=True))(x)
    x = Dropout(0.25)(x)
    x = Attention(return_sequences=False)(x)

    rnn_model = get_model(learning_rate, optimizer, features_input, x)

    return rnn_model


def get_model(learning_rate, optimizer, features_input, layer):
    output = Dense(NUMBER_OF_CLASSES, activation='softmax')(layer)

    rnn_model = keras.Model([features_input], output)

    if optimizer == 'adam':
        rnn_model.compile(
            loss='sparse_categorical_crossentropy', optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate), metrics=['accuracy']
        )
    else:
        rnn_model.compile(
            loss='sparse_categorical_crossentropy', optimizer=tf.keras.optimizers.SGD(learning_rate=learning_rate, momentum=0.9), metrics=['accuracy']
        )

    return rnn_model
