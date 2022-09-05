import os
import time

import keras_tuner as kt
import tensorflow as tf
from tensorflow.keras.layers import Bidirectional, LSTM, GRU, Dropout, Dense
from tensorflow import keras
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Concatenate, TimeDistributed

from utils import cnn_models
from utils import rnn_models

NUMBER_OF_CLASSES = 226
HAND_WIDTH, HAND_HEIGHT = 100, 200
FACE_WIDTH, FACE_HEIGHT = 100, 100


def get_hand_sequence(hand_input, fine_tune):
    # merged = Concatenate()([input_1, input_2])
    cnn_model = cnn_models.get_efficientnet_model(
        hand_input, prefix_name='hand', fine_tune=fine_tune)

    return cnn_model


def get_cnn_model(fine_tune=False):
    hand_input = tf.keras.layers.Input(
        shape=(HAND_WIDTH, HAND_HEIGHT, 3), name='hand_input')

    hand_seq = get_hand_sequence(hand_input, fine_tune)

    model = Model(inputs=[hand_input], outputs=hand_seq)
    return model


def face_model_builder(hp):
    n_layers = hp.Int(name='face_n_layers',
                      min_value=1, max_value=3, step=1)
    rnn_type = hp.Choice(name='face_rnn_type', values=[
                         'lstm', 'gru'], ordered=False)

    dropout_rate = hp.Float(
        name='face_dropout', min_value=0.05, max_value=0.40, step=0.05)

    bidirectional = hp.Boolean(name='face_bidirectional')
    attention = hp.Boolean(name='face_attention')

    RNN = GRU if rnn_type == 'gru' else LSTM

    face_input = [keras.Input((16, 138), name='face_data')]

    for layer in range(1, n_layers+1):
        hp_units = hp.Int(
            f'face_hp_units_{layer}', min_value=32, max_value=480, step=64)

        return_seq = False if n_layers == layer and not attention else True

        if layer == 1:
            y = add_rnn_layer(bidirectional, hp_units,
                              return_seq, RNN, face_input)
        else:
            y = add_rnn_layer(bidirectional, hp_units, return_seq, RNN, y)

        if layer in [1, 2]:
            y = Dropout(dropout_rate)(y)

    if attention:
        y = rnn_models.Attention(return_sequences=False)(y)

    model = Model(inputs=[face_input], outputs=y)
    return model


def triangle_model_builder(hp):
    n_layers = hp.Int(name='triangle_n_layers',
                      min_value=1, max_value=3, step=1)
    rnn_type = hp.Choice(name='triangle_rnn_type', values=[
                         'lstm', 'gru'], ordered=False)

    dropout_rate = hp.Float(
        name='triangle_dropout', min_value=0.05, max_value=0.40, step=0.05)

    bidirectional = hp.Boolean(name='triangle_bidirectional')
    attention = hp.Boolean(name='triangle_attention')

    RNN = GRU if rnn_type == 'gru' else LSTM

    tri_input = [keras.Input((16, 13), name='triangle_data')]

    for layer in range(1, n_layers+1):
        hp_units = hp.Int(
            f'triangle_hp_units_{layer}', min_value=32, max_value=320, step=64)

        return_seq = False if n_layers == layer and not attention else True

        if layer == 1:
            y = add_rnn_layer(bidirectional, hp_units,
                              return_seq, RNN, tri_input)
        else:
            y = add_rnn_layer(bidirectional, hp_units, return_seq, RNN, y)

        if layer in [1, 2]:
            y = Dropout(dropout_rate)(y)

    if attention:
        y = rnn_models.Attention(return_sequences=False)(y)

    model = Model(inputs=[tri_input], outputs=y)
    return model


def add_rnn_layer(bidirectional, hp_units, return_seq, RNN, y):
    if bidirectional:
        y = Bidirectional(RNN(hp_units, return_sequences=return_seq))(y)
    else:
        y = RNN(hp_units, return_sequences=return_seq)(y)

    return y


def rnn_cnn_model_builder(hp):
    n_layers = hp.Int(name='rnn_cnn_n_layers',
                      min_value=1, max_value=5, step=1)

    rnn_type = hp.Choice(name='rnn_cnn_rnn_type', values=[
                         'lstm', 'gru'], ordered=False)

    bidirectional = hp.Boolean(name='rnn_cnn_bidirectional')
    attention = hp.Boolean(name='rnn_cnn_attention')

    RNN = GRU if rnn_type == 'gru' else LSTM

    dropout_rate = hp.Float(
        name='rnn_cnn_dropout_1', min_value=0.05, max_value=0.40, step=0.05)

    frame_features_input = [keras.Input(
        (16, 100, 200, 3), name="input"+str(c)) for c in range(1)]

    cnn_model = get_cnn_model(False)
    y = TimeDistributed(cnn_model)(frame_features_input)

    for layer in range(1, n_layers+1):
        hp_units = hp.Int(
            f'rnn_cnn_units_{layer}', min_value=32, max_value=512, step=64)

        return_seq = False if n_layers == layer and not attention else True
        y = add_rnn_layer(bidirectional, hp_units, return_seq, RNN, y)

        if layer in [1, 2, 3]:
            y = Dropout(dropout_rate)(y)

    if attention:
        y = rnn_models.Attention(return_sequences=False)(y)

    model = Model(inputs=[frame_features_input], outputs=y)
    return model


def get_meta_learner(hp, concat_layers):
    with hp.conditional_scope("join_type", ['meta_learner']):
        n_layers = hp.Int(name='join_n_layers',
                          min_value=1, max_value=2, step=1)
        hp_units = hp.Int('join_units_1', min_value=32, max_value=512, step=64)

        dropout_rate = hp.Float(
            name='join_dropout_1', min_value=0.05, max_value=0.35, step=0.05)

        dense = Dense(hp_units, activation='elu')(concat_layers)
        dense = Dropout(dropout_rate)(dense)

        with hp.conditional_scope("join_n_layers", [2]):
            if n_layers >= 2:
                hp_units_2 = hp.Int('join_units_2', min_value=32, max_value=512,
                                    step=64)
                dense = Dense(hp_units_2, activation='elu')(dense)

        return dense


def join_archtectures(hp, cnn_rnn_layer, triangle_rnn_layer, face_rnn_layer):
    join_type = hp.Choice(name='join_type', values=[
        'multi_classifier', 'single_classifier', 'meta_learner'], ordered=False)

    if join_type == 'multi_classifier':
        classifier_1 = Dense(
            NUMBER_OF_CLASSES, activation='softmax')(cnn_rnn_layer)
        classifier_2 = Dense(NUMBER_OF_CLASSES, activation='softmax')(
            triangle_rnn_layer)
        classifier_3 = Dense(
            NUMBER_OF_CLASSES, activation='softmax')(face_rnn_layer)
        output = tf.keras.layers.Average()(
            [classifier_1, classifier_2, classifier_3])
    elif join_type == 'single_classifier':
        concat_layers = Concatenate()(
            [cnn_rnn_layer, triangle_rnn_layer, face_rnn_layer])
        output = Dense(NUMBER_OF_CLASSES, activation='softmax')(concat_layers)
    elif join_type == 'meta_learner':
        concat_layers = Concatenate()(
            [cnn_rnn_layer, triangle_rnn_layer, face_rnn_layer])
        meta_learner = get_meta_learner(hp, concat_layers)
        output = Dense(NUMBER_OF_CLASSES, activation='softmax')(meta_learner)

    return output


def model_builder(hp):
    cnn_rnn_layer = rnn_cnn_model_builder(hp)
    triangle_rnn_layer = triangle_model_builder(hp)
    face_rnn_layer = face_model_builder(hp)

    output = join_archtectures(
        hp, cnn_rnn_layer.output, triangle_rnn_layer.output, face_rnn_layer.output)

    rnn_model = keras.Model(
        [cnn_rnn_layer.input, triangle_rnn_layer.input, face_rnn_layer.input], output)

    rnn_model.compile(
        loss='sparse_categorical_crossentropy', optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3), metrics=['accuracy']
    )

    return rnn_model


def get_tuner_instance():
    tuner = kt.Hyperband(hypermodel=model_builder,
                         objective=kt.Objective(
                             "val_accuracy", direction="max"),
                         max_epochs=15,
                         project_name='tuner_results/hyperband_tuner_new_join_hands')

    print(tuner.search_space_summary())

    return tuner
