import keras_tuner as kt
import tensorflow as tf
from tensorflow.keras.layers import Bidirectional, LSTM, GRU, Dropout, Dense
from tensorflow import keras
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Concatenate, TimeDistributed

from utils import cnn_models
from utils import rnn_models

NUMBER_OF_CLASSES = 226
HAND_WIDTH, HAND_HEIGHT = 80, 80
FACE_WIDTH, FACE_HEIGHT = 80, 80


def get_hand_sequence(input_1, input_2, fine_tune):
    merged = Concatenate()([input_1, input_2])
    cnn_model = cnn_models.get_efficientnet_model(
        merged, prefix_name='hand', fine_tune=fine_tune)

    return cnn_model


def get_cnn_model(fine_tune=False):
    hand1_input = tf.keras.layers.Input(
        shape=(HAND_WIDTH, HAND_HEIGHT, 3), name='hand1_input')
    hand2_input = tf.keras.layers.Input(
        shape=(HAND_WIDTH, HAND_HEIGHT, 3), name='hand2_input')
    face_input = tf.keras.layers.Input(
        shape=(FACE_WIDTH, FACE_HEIGHT, 3), name='face_input')
    triangle_input = tf.keras.layers.Input(shape=(11), name='triangle_input')

    hand_seq = get_hand_sequence(hand1_input, hand2_input, fine_tune)
    # face_seq = get_face_sequence(face_input, fine_tune)

    # concat_layers = Concatenate()([hand_seq, face_seq])
    # final_output = Concatenate()([hand_seq, triangle_input])

    model = Model(inputs=[hand1_input, hand2_input], outputs=hand_seq)
    tf.keras.utils.plot_model(model, "model.png", show_shapes=True)

    return model


def triangle_model_builder(hp):
    n_layers = hp.Int(name='n_layers', min_value=1, max_value=4, step=1)
    rnn_type = hp.Choice(name='rnn_type', values=[
                         'lstm', 'gru'], ordered=False)
    hp_units = hp.Int('units', min_value=32, max_value=256, step=64)
    hp_units_2 = hp.Int('hp_units_2', min_value=32, max_value=256, step=64)
    hp_units_3 = hp.Int('hp_units_3', min_value=32, max_value=256, step=64)
    hp_units_4 = hp.Int('hp_units_4', min_value=32, max_value=256, step=64)
    dropout_rate = hp.Float(
        name='dropout', min_value=0.1, max_value=0.5, step=0.05)

    RNN = GRU if rnn_type == 'gru' else LSTM

    tri_input = [keras.Input((16, 11), name='triangle_data')]

    return_seq = False if n_layers == 1 else True
    y = RNN(hp_units, return_sequences=return_seq)(tri_input)
    y = Dropout(dropout_rate)(y)

    if n_layers >= 2:
        return_seq = False if n_layers == 2 else True

        y = RNN(hp_units_2, return_sequences=return_seq)(y)
        y = Dropout(dropout_rate)(y)
    if n_layers >= 3:
        return_seq = False if n_layers == 3 else True
        y = RNN(hp_units_3, return_sequences=return_seq)(y)
        y = Dropout(dropout_rate)(y)
    if n_layers >= 4:
        y = RNN(hp_units_4, return_sequences=False)(y)
        y = Dropout(dropout_rate)(y)

    output = Dense(NUMBER_OF_CLASSES, activation='softmax')(y)

    rnn_model = keras.Model([tri_input], output)

    rnn_model.compile(
        loss='sparse_categorical_crossentropy', optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3), metrics=['accuracy']
    )

    return rnn_model


def rnn_cnn_model_builder(hp):
    cnn_model = get_cnn_model(False)
    n_layers = hp.Int(name='n_layers', min_value=1, max_value=4, step=1)

    rnn_type = hp.Choice(name='rnn_type', values=[
                         'lstm', 'gru'], ordered=False)

    bidirectional = hp.Boolean(name='bidirectional')
    attention = hp.Boolean(name='attention')

    hp_units = hp.Int('units_1', min_value=32, max_value=512, step=64)
    hp_units_2 = hp.Int('units_2', min_value=32, max_value=512, step=64)
    hp_units_3 = hp.Int('units_3', min_value=32, max_value=512, step=64)
    hp_units_4 = hp.Int('units_4', min_value=32, max_value=512, step=64)
    RNN = GRU if rnn_type == 'gru' else LSTM

    dropout_rate = hp.Float(
        name='dropout_1', min_value=0.1, max_value=0.5, step=0.05)

    frame_features_input = [keras.Input(
        (16, 80, 80, 3), name="input"+str(c)) for c in range(2)]

    y = TimeDistributed(cnn_model)(frame_features_input)

    return_seq = False if n_layers == 1 and not attention else True
    if bidirectional:
        y = Bidirectional(RNN(hp_units, return_sequences=return_seq))(y)
    else:
        y = RNN(hp_units, return_sequences=return_seq)(y)

    y = Dropout(dropout_rate)(y)

    if n_layers >= 2:
        return_seq = False if n_layers == 2 and not attention else True

        if bidirectional:
            y = Bidirectional(RNN(hp_units_2, return_sequences=return_seq))(y)
        else:
            y = RNN(hp_units_2, return_sequences=return_seq)(y)
        y = Dropout(dropout_rate)(y)

    if n_layers >= 3:
        return_seq = False if n_layers == 3 and not attention else True
        if bidirectional:
            y = Bidirectional(RNN(hp_units_3, return_sequences=return_seq))(y)
        else:
            y = RNN(hp_units_3, return_sequences=return_seq)(y)
        y = Dropout(dropout_rate)(y)
    if n_layers >= 4:
        return_seq = False if not attention else True
        if bidirectional:
            y = Bidirectional(RNN(hp_units_4, return_sequences=return_seq))(y)
        else:
            y = RNN(hp_units_4, return_sequences=return_seq)(y)

    if attention:
        y = rnn_models.Attention(return_sequences=False)(y)

    output = Dense(NUMBER_OF_CLASSES, activation='softmax')(y)

    rnn_model = keras.Model([frame_features_input], output)

    rnn_model.compile(
        loss='sparse_categorical_crossentropy', optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3), metrics=['accuracy']
    )

    return rnn_model


def get_tuner_instance():
    tuner = kt.Hyperband(hypermodel=rnn_cnn_model_builder,
                         objective=kt.Objective(
                             "val_accuracy", direction="max"),
                         max_epochs=15,
                         project_name='hyperband_tuner_cnn_lstm')

    print(tuner.search_space_summary())

    return tuner
