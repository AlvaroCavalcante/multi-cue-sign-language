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

    return model


def triangle_model_builder(hp):
    n_layers = hp.Int(name='triangle_n_layers',
                      min_value=1, max_value=3, step=1)
    rnn_type = hp.Choice(name='triangle_rnn_type', values=[
                         'lstm', 'gru'], ordered=False)

    dropout_rate = hp.Float(
        name='triangle_dropout', min_value=0.1, max_value=0.40, step=0.05)

    bidirectional = hp.Boolean(name='triangle_bidirectional')
    attention = hp.Boolean(name='triangle_attention')

    RNN = GRU if rnn_type == 'gru' else LSTM

    tri_input = [keras.Input((16, 11), name='triangle_data')]

    for layer in range(1, n_layers+1):
        hp_units = hp.Int(f'triangle_hp_units_{layer}', min_value=32, max_value=256, step=64)

        return_seq = False if n_layers == layer and not attention else True
        
        if layer == 1:
            y = add_rnn_layer(bidirectional, hp_units, return_seq, RNN, tri_input)
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
                      min_value=1, max_value=4, step=1)

    rnn_type = hp.Choice(name='rnn_cnn_rnn_type', values=[
                         'lstm', 'gru'], ordered=False)

    bidirectional = hp.Boolean(name='rnn_cnn_bidirectional')
    attention = hp.Boolean(name='rnn_cnn_attention')

    RNN = GRU if rnn_type == 'gru' else LSTM

    dropout_rate = hp.Float(
        name='rnn_cnn_dropout_1', min_value=0.1, max_value=0.40, step=0.05)

    frame_features_input = [keras.Input(
        (16, 80, 80, 3), name="input"+str(c)) for c in range(2)]

    cnn_model = get_cnn_model(False)
    y = TimeDistributed(cnn_model)(frame_features_input)

    for layer in range(1, n_layers+1):
        hp_units = hp.Int(f'rnn_cnn_units_{layer}', min_value=32, max_value=512, step=64)

        return_seq = False if n_layers == layer and not attention else True
        y = add_rnn_layer(bidirectional, hp_units, return_seq, RNN, y)

        if layer in [1, 2, 3]:
            y = Dropout(dropout_rate)(y)

    if attention:
        y = rnn_models.Attention(return_sequences=False)(y)

    model = Model(inputs=[frame_features_input], outputs=y)
    return model


def get_meta_learner(hp, concat_layers):
    n_layers = hp.Int(name='join_n_layers', min_value=1, max_value=2, step=1)
    hp_units = hp.Int('join_units_1', min_value=32, max_value=512, step=64)
    hp_units_2 = hp.Int('join_units_2', min_value=32, max_value=512,
                        step=64)

    dropout_rate = hp.Float(
        name='join_dropout_1', min_value=0.1, max_value=0.40, step=0.05)

    dense = Dense(hp_units, activation='elu')(concat_layers)
    dense = Dropout(dropout_rate)(dense)

    with hp.conditional_scope("join_n_layers", [2]):
        if n_layers >= 2:
            dense = Dense(hp_units_2, activation='elu')(dense)

    return dense


def join_archtectures(hp, cnn_rnn_layer, triangle_rnn_layer):
    join_type = hp.Choice(name='join_type', values=[
        'multi_classifier', 'single_classifier', 'meta_learner'], ordered=False)

    if join_type == 'multi_classifier':
        output1 = Dense(NUMBER_OF_CLASSES, activation='softmax')(cnn_rnn_layer)
        output2 = Dense(NUMBER_OF_CLASSES, activation='softmax')(
            triangle_rnn_layer)
        output = tf.keras.layers.Average()([output1, output2])
    elif join_type == 'single_classifier':
        concat_layers = Concatenate()([cnn_rnn_layer, triangle_rnn_layer])
        output = Dense(NUMBER_OF_CLASSES, activation='softmax')(concat_layers)
    else:
        concat_layers = Concatenate()([cnn_rnn_layer, triangle_rnn_layer])
        meta_learner = get_meta_learner(hp, concat_layers)
        output = Dense(NUMBER_OF_CLASSES, activation='softmax')(meta_learner)

    return output


def model_builder(hp):
    cnn_rnn_layer = rnn_cnn_model_builder(hp)
    triangle_rnn_layer = triangle_model_builder(hp)
    output = join_archtectures(
        hp, cnn_rnn_layer.output, triangle_rnn_layer.output)

    rnn_model = keras.Model(
        [cnn_rnn_layer.input, triangle_rnn_layer.input], output)

    rnn_model.compile(
        loss='sparse_categorical_crossentropy', optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3), metrics=['accuracy']
    )

    return rnn_model


def get_tuner_instance():
    tuner = kt.Hyperband(hypermodel=model_builder,
                         objective=kt.Objective(
                             "val_accuracy", direction="max"),
                         max_epochs=15,
                         project_name='hyperband_tuner_cnn_lstm')

    print(tuner.search_space_summary())

    return tuner
