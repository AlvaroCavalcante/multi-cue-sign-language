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
HAND_WIDTH, HAND_HEIGHT = 100, 100
FACE_WIDTH, FACE_HEIGHT = 100, 100


def get_recurrent_model(learning_rate, cnn_model):
    frame_features_input = [keras.Input(
        (16, HAND_WIDTH, HAND_HEIGHT, 3), name="input"+str(c)) for c in range(2)]

    x = TimeDistributed(cnn_model)(frame_features_input)
    x = GRU(480, return_sequences=True)(x)
    x = Dropout(0.35)(x)
    x = GRU(288, return_sequences=True)(x)
    x = Dropout(0.35)(x)
    x = GRU(288, return_sequences=True)(x)
    x = Dropout(0.35)(x)
    x = rnn_models.Attention(return_sequences=False)(x)

    tri_input = [keras.Input((16, 13), name='triangle_data')]
    y = GRU(224, return_sequences=True)(tri_input)
    y = Dropout(0.20)(y)
    y = GRU(96, return_sequences=True)(y)
    y = rnn_models.Attention(return_sequences=False)(y)

    face_input = [keras.Input((16, 138), name='face_input')]
    z = GRU(448, return_sequences=True)(face_input)
    z = Dropout(0.15)(z)
    z = GRU(224, return_sequences=True)(z)
    z = Dropout(0.15)(z)
    z = rnn_models.Attention(return_sequences=False)(z)

    # output1 = Dense(NUMBER_OF_CLASSES, activation='softmax')(y)
    # output2 = Dense(NUMBER_OF_CLASSES, activation='softmax')(x)

    concat_layers = Concatenate()([x, y, z])

    # output = tf.keras.layers.Average()([output1, output1])

    dense = Dense(512, activation='elu')(concat_layers)
    dense = Dropout(0.15)(dense)
    dense = Dense(256, activation='elu')(dense)
    output = Dense(NUMBER_OF_CLASSES, activation='softmax')(dense)

    rnn_model = keras.Model([frame_features_input, tri_input, face_input], output)

    rnn_model.compile(
        loss='sparse_categorical_crossentropy', optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate), metrics=['accuracy']
    )

    # tf.keras.utils.plot_model(rnn_model, "model.png", show_shapes=True)
    print(rnn_model.summary())
    return rnn_model


def get_hand_sequence(input_1, input_2, fine_tune):
    merged = Concatenate()([input_1, input_2])
    cnn_model = cnn_models.get_efficientnet_model(
        merged, prefix_name='hand', fine_tune=fine_tune)

    return cnn_model


def get_face_sequence(face_input, fine_tune):
    cnn_model = cnn_models.get_efficientnet_model(
        face_input, prefix_name='face', fine_tune=fine_tune)
    return cnn_model


def get_cnn_model(fine_tune=False):
    hand1_input = tf.keras.layers.Input(
        shape=(HAND_WIDTH, HAND_HEIGHT, 3), name='hand1_input')
    hand2_input = tf.keras.layers.Input(
        shape=(HAND_WIDTH, HAND_HEIGHT, 3), name='hand2_input')
    # face_input = tf.keras.layers.Input(
    #     shape=(FACE_WIDTH, FACE_HEIGHT, 3), name='face_input')

    hand_seq = get_hand_sequence(hand1_input, hand2_input, fine_tune)
    # face_seq = get_face_sequence(face_input, fine_tune)

    # concat_layers = Concatenate()([hand_seq, face_seq])

    model = Model(inputs=[hand1_input, hand2_input], outputs=hand_seq)
    tf.keras.utils.plot_model(model, "model.png", show_shapes=True)

    return model


def train_gen(dataset):
    for (hand_1, hand_2, triangle_data, face_keypoints, label) in dataset:
        yield [hand_1, hand_2, triangle_data, face_keypoints], label


def eval_gen(dataset):
    for (data, label) in dataset:
        yield data, label


def train_cnn_lstm_model(train_files, eval_files, epochs, batch_size, learning_rate, load_weights=False, tune_model=False):
    dataset = load_data_tfrecord(train_files, batch_size)
    dataset_eval = load_data_tfrecord(eval_files, batch_size, False)

    train_steps, val_steps = utils.get_steps(
        train_files, eval_files, batch_size)

    logdir = "src/logs/scalars/" + datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = keras.callbacks.TensorBoard(log_dir=logdir)

    early_stop = EarlyStopping(monitor="val_loss", patience=3)

    callbacks_list = [
        ModelCheckpoint('/home/alvaro/Desktop/multi-cue-sign-language/src/models/tunned_model/', monitor='val_accuracy',
                        verbose=1, save_best_only=True, save_weights_only=True),
        # LearningRateScheduler(lr_scheduler.lr_time_based_decay, verbose=1),
        tensorboard_callback,
        # early_stop
    ]

    if tune_model:
        print('Training model using keras tuner')
        tuner = model_tuner.get_tuner_instance()
        tuner.search(x=train_gen(dataset),
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
                '/home/alvaro/Desktop/multi-cue-sign-language/src/models/cnn_lstm_v2_fine_5/').expect_partial()

        print('Training model')
        recurrent_model.fit(train_gen(dataset),
                            steps_per_epoch=train_steps,
                            epochs=epochs,
                            validation_data=eval_gen(dataset_eval),
                            validation_steps=val_steps,
                            callbacks=callbacks_list)


if __name__ == '__main__':
    train_files = tf.io.gfile.glob(
        '/home/alvaro/Desktop/video2tfrecord/example/train_v2_edited/*.tfrecords')

    eval_files = tf.io.gfile.glob(
        '/home/alvaro/Desktop/video2tfrecord/example/val_v2_edited/*.tfrecords')

    epochs = 25
    batch_size = 30
    learning_rate = 0.0001
    train_cnn_lstm_model(train_files, eval_files, epochs,
                         batch_size, learning_rate, False, True)
