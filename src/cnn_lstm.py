from datetime import datetime

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Concatenate, TimeDistributed
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler, EarlyStopping
from tensorflow.keras.layers import Dropout, Dense

from read_dataset import load_data_tfrecord
from utils import utils
from utils import lr_scheduler
from utils import cnn_models
from utils import rnn_models
from utils import model_tuner

physical_devices = tf.config.list_physical_devices('GPU')
try:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
except:
    print('GPU not found')


MAX_SEQ_LENGTH = 16
NUMBER_OF_CLASSES = 500
HAND_WIDTH, HAND_HEIGHT = 100, 200
FACE_WIDTH, FACE_HEIGHT = 100, 100
TRIANGLE_FIG_WIDTH, TRIANGLE_FIG_HEIGHT = 128, 128


def get_recurrent_model(learning_rate):
    # face_cnn = get_cnn_model(FACE_WIDTH, FACE_HEIGHT, 'face')
    # face_model = rnn_models.get_face_rnn_model(face_cnn, learning_rate)

    # hands_cnn = get_cnn_model(HAND_WIDTH, HAND_HEIGHT, 'hands', fine_tune=True)
    # hands_model = rnn_models.get_hands_rnn_model(hands_cnn, learning_rate, optimizer='SGD')

    triangle_model = rnn_models.get_triangle_rnn_model(learning_rate)

    # triangle_fig_model = rnn_models.get_triangle_figure_rnn_model(
    #     cnn_model, learning_rate)

    # triangle_fig_model.load_weights(
    #     '/home/alvaro/Desktop/multi-cue-sign-language/src/models/new_tri_fig_mobile_fine_v1/').expect_partial()

    # face_model.load_weights(
    #     '/home/alvaro/Desktop/multi-cue-sign-language/src/models/step1_face_fine_v4/').expect_partial()

    # hands_model.load_weights(
    #     '/home/alvaro/Desktop/multi-cue-sign-language/src/models/step1_hands_csl_fine/').expect_partial()

    # concat_layers = Concatenate()([
    #     hands_model.layers[-2].output, triangle_model.layers[-2].output, face_model.layers[-2].output])

    # output = Dense(NUMBER_OF_CLASSES, activation='softmax')(hands_model.layers[-2].output)

    # rnn_model = keras.Model(
    #     [hands_model.input], output)

    # rnn_model.compile(
    #     loss='sparse_categorical_crossentropy', optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate), metrics=['accuracy']
    # )

    print(triangle_model.summary())
    return triangle_model


def get_cnn_model(width: int, height: int, prefix_name: str, fine_tune=False):
    model_input = tf.keras.layers.Input(
        shape=(width, height, 3), name=f'{prefix_name}_input')

    cnn_model = cnn_models.get_efficientnet_model(
        model_input, prefix_name=prefix_name, fine_tune=fine_tune)

    model = Model(inputs=[model_input], outputs=cnn_model)
    # tf.keras.utils.plot_model(model, "model_plot.png", show_shapes=True)
    return model


def train_gen(dataset):
    for (data, label) in dataset:
        yield data, label


def eval_gen(dataset):
    for (data, label) in dataset:
        yield data, label


def train_cnn_lstm_model(train_files, eval_files, epochs, batch_size, learning_rate, load_weights=False, tune_model=False, train_tuned_model=False):
    dataset = load_data_tfrecord(train_files, batch_size)
    dataset_eval = load_data_tfrecord(eval_files, batch_size, False)

    train_steps, val_steps = utils.get_steps(
        train_files, eval_files, batch_size)

    logdir = "src/logs/scalars/" + datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = keras.callbacks.TensorBoard(log_dir=logdir)

    callbacks_list = [
        ModelCheckpoint('/home/alvaro/Desktop/multi-cue-sign-language/src/models/step1_triangle_csl/', monitor='val_accuracy',
                        verbose=1, save_best_only=True, save_weights_only=True),
        # LearningRateScheduler(lr_scheduler.lr_asc_desc_decay, verbose=1),
        tensorboard_callback,
        # EarlyStopping(monitor="val_loss", patience=3)
    ]

    if tune_model:
        print('Training model using keras tuner')
        tuner = model_tuner.get_tuner_instance()
        tuner.results_summary()
        tuner.search_space_summary()
        tuner.search(x=train_gen(dataset),
                     steps_per_epoch=train_steps,
                     epochs=epochs,
                     validation_data=eval_gen(dataset_eval),
                     validation_steps=val_steps,
                     callbacks=callbacks_list)
        # best_model = tuner.get_best_models()[0]
    elif train_tuned_model:
        tuner = model_tuner.get_tuner_instance()
        best_hp = tuner.get_best_hyperparameters(num_trials=1)[0]
        model = tuner.hypermodel.build(best_hp)
        tf.keras.utils.plot_model(model, "model_plot.png", show_shapes=True)
        model.fit(train_gen(dataset),
                  steps_per_epoch=train_steps,
                  epochs=epochs,
                  validation_data=eval_gen(dataset_eval),
                  validation_steps=val_steps,
                  callbacks=callbacks_list)
    else:
        recurrent_model = get_recurrent_model(learning_rate)

        if load_weights:
            recurrent_model.load_weights(
                '/home/alvaro/Desktop/multi-cue-sign-language/src/models/new_tri_fig_mobile/').expect_partial()

        print('Training model')
        recurrent_model.fit(train_gen(dataset),
                            steps_per_epoch=train_steps,
                            epochs=epochs,
                            validation_data=eval_gen(dataset_eval),
                            validation_steps=val_steps,
                            callbacks=callbacks_list)


if __name__ == '__main__':
    train_files = tf.io.gfile.glob(
        '/home/alvaro/Desktop/video2tfrecord/results/train/*.tfrecords')

    eval_files = tf.io.gfile.glob(
        '/home/alvaro/Desktop/video2tfrecord/results/test/*.tfrecords')

    epochs = 40
    batch_size = 30
    learning_rate = 1e-3
    train_cnn_lstm_model(train_files, eval_files, epochs,
                         batch_size, learning_rate,
                         load_weights=False,
                         tune_model=False,
                         train_tuned_model=False
                         )
