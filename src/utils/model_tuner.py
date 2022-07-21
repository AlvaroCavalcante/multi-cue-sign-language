import keras_tuner as kt
from tensorflow.keras.layers import Bidirectional, LSTM, GRU, Dropout, Dense
from tensorflow import keras

NUMBER_OF_CLASSES = 226


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

    tri_input = [keras.Input((16, 11), name='triangle_data')]

    RNN = GRU if rnn_type == 'gru' else LSTM

    return_seq = False if n_layers == 1 else True
    y = RNN(hp_units, return_sequences=return_seq)(tri_input)
    y = Dropout(dropout_rate)(y)

    if n_layers >= 2:
        return_seq = False if n_layers == 2 else True

        y = RNN(hp_units_2, return_sequences=return_seq)(tri_input)
        y = Dropout(dropout_rate)(y)
    if n_layers >= 3:
        return_seq = False if n_layers == 3 else True
        y = RNN(hp_units_3, return_sequences=return_seq)(tri_input)
        y = Dropout(dropout_rate)(y)
    if n_layers >= 4:
        y = RNN(hp_units_4, return_sequences=False)(tri_input)
        y = Dropout(dropout_rate)(y)

    output = Dense(NUMBER_OF_CLASSES, activation='softmax')(y)

    rnn_model = keras.Model([tri_input], output)

    rnn_model.compile(
        loss='sparse_categorical_crossentropy', optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3), metrics=['accuracy']
    )

    return rnn_model


def get_tuner_instance():
    tuner = kt.Hyperband(hypermodel=triangle_model_builder,
                         objective=kt.Objective(
                             "val_accuracy", direction="max"),
                         max_epochs=15,
                         project_name='hyperband_tuner_triangle')

    print(tuner.search_space_summary())

    return tuner
