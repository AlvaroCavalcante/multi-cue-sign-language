from datetime import datetime

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Concatenate, TimeDistributed
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler, EarlyStopping
from tensorflow.keras.layers import Bidirectional, LSTM, GRU, Dropout, Dense

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
NUMBER_OF_CLASSES = 226
HAND_WIDTH, HAND_HEIGHT = 100, 200
FACE_WIDTH, FACE_HEIGHT = 100, 100


def get_recurrent_model(learning_rate, cnn_model):
    frame_features_input = [keras.Input(
        (16, HAND_WIDTH, HAND_HEIGHT, 3), name="input"+str(c)) for c in range(1)]

    x = TimeDistributed(cnn_model)(frame_features_input)
    x = Bidirectional(GRU(288, return_sequences=True))(x)
    x = Dropout(0.25)(x)
    x = rnn_models.Attention(return_sequences=False)(x)

    output = Dense(NUMBER_OF_CLASSES, activation='softmax')(x)

    rnn_model = keras.Model(
        [frame_features_input], output)

    rnn_model.compile(
        loss='sparse_categorical_crossentropy', optimizer=tf.keras.optimizers.SGD(learning_rate=learning_rate, momentum=0.9), metrics=['accuracy'] # , momentum=0.9
    )

    print(rnn_model.summary())
    return rnn_model


def get_hand_sequence(hand_input, fine_tune):
    cnn_model = cnn_models.get_efficientnet_model(
        hand_input, prefix_name='hand', fine_tune=fine_tune)

    return cnn_model


def get_face_sequence(face_input, fine_tune):
    cnn_model = cnn_models.get_efficientnet_model(
        face_input, prefix_name='face', fine_tune=fine_tune)
    return cnn_model


def get_cnn_model(fine_tune=False):
    hand_input = tf.keras.layers.Input(
        shape=(HAND_WIDTH, HAND_HEIGHT, 3), name='hand_input')

    hand_seq = get_hand_sequence(hand_input, fine_tune)

    model = Model(inputs=[hand_input], outputs=hand_seq)
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
        ModelCheckpoint('/home/alvaro/Desktop/multi-cue-sign-language/src/models/step1_triangle/', monitor='val_accuracy',
                        verbose=1, save_best_only=True, save_weights_only=True),
        # LearningRateScheduler(lr_scheduler.lr_asc_desc_decay, verbose=1),
        tensorboard_callback,
        # EarlyStopping(monitor="val_loss", patience=3)
    ]

    if tune_model:
        print('Training model using keras tuner')
        tuner = model_tuner.get_tuner_instance()
        tuner.results_summary()
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
        cnn_model = get_cnn_model(load_weights)
        recurrent_model = get_recurrent_model(learning_rate, cnn_model)

        if load_weights:
            recurrent_model.load_weights(
                '/home/alvaro/Desktop/multi-cue-sign-language/src/models/step1_hands_fine_v3/').expect_partial()

        print('Training model')
        recurrent_model.fit(train_gen(dataset),
                            steps_per_epoch=train_steps,
                            epochs=epochs,
                            validation_data=eval_gen(dataset_eval),
                            validation_steps=val_steps,
                            callbacks=callbacks_list)


if __name__ == '__main__':
    train_files = tf.io.gfile.glob( 
        '/home/alvaro/Desktop/video2tfrecord/results/train_v5/*.tfrecords')

    eval_files = tf.io.gfile.glob(
        '/home/alvaro/Desktop/video2tfrecord/results/val_v5/*.tfrecords')

    epochs = 40
    batch_size = 30
    learning_rate = 1e-3
    train_cnn_lstm_model(train_files, eval_files, epochs,
                         batch_size, learning_rate, False, False, True)
