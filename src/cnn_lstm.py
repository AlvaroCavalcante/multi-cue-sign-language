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

physical_devices = tf.config.list_physical_devices('GPU')
try:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
except:
    print('GPU not found')


MAX_SEQ_LENGTH = 16
NUMBER_OF_CLASSES = 226
HAND_WIDTH, HAND_HEIGHT = 80, 80
FACE_WIDTH, FACE_HEIGHT = 80, 80


def get_recurrent_model(learning_rate, cnn_model):
    frame_features_input = [keras.Input(
        (16, HAND_WIDTH, HAND_HEIGHT, 3), name="input"+str(c)) for c in range(2)]
    frame_features_input.append(keras.Input((16, 11), name='triangle_data'))

    x = TimeDistributed(cnn_model)(frame_features_input)
    x = Bidirectional(GRU(512, return_sequences=True))(x)
    x = Dropout(0.20)(x)
    x = Bidirectional(GRU(256, return_sequences=True))(x)
    x = Dropout(0.20)(x)
    x = Bidirectional(GRU(256, return_sequences=True))(x)
    # x = Dense(64, activation='elu')(x)
    x = rnn_models.Attention(return_sequences=False)(x)
    output = Dense(NUMBER_OF_CLASSES, activation='softmax')(x)

    rnn_model = keras.Model(frame_features_input, output)

    rnn_model.compile(
        loss='sparse_categorical_crossentropy', optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate), metrics=['accuracy']
    )

    # tf.keras.utils.plot_model(rnn_model, "model.png", show_shapes=True)
    print(rnn_model.summary())
    return rnn_model


def get_hand_sequence(input_1, input_2, fine_tune):
    merged = Concatenate()([input_1, input_2])
    cnn_model = cnn_models.get_efficientnet_v2_model(
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
    face_input = tf.keras.layers.Input(
        shape=(FACE_WIDTH, FACE_HEIGHT, 3), name='face_input')
    triangle_input = tf.keras.layers.Input(shape=(11), name='triangle_input')

    hand_seq = get_hand_sequence(hand1_input, hand2_input, fine_tune)
    # face_seq = get_face_sequence(face_input, fine_tune)

    # concat_layers = Concatenate()([hand_seq, face_seq])
    final_output = Concatenate()([hand_seq, triangle_input])

    model = Model(inputs=[hand1_input, hand2_input,
                  triangle_input], outputs=final_output)
    tf.keras.utils.plot_model(model, "model.png", show_shapes=True)

    return model


def train_gen(dataset):
    for (hand_seq, triangle_data, centroids, label, video_name_list, triangle_stream) in dataset:
        yield [hand_seq[:, 0], hand_seq[:, 1], triangle_data], label


def eval_gen(dataset):
    for (data, label, video_name) in dataset:
        yield data, label


def train_cnn_lstm_model(train_files, eval_files, epochs, batch_size, learning_rate, load_weights=False):
    dataset = load_data_tfrecord(train_files, batch_size)
    dataset_eval = load_data_tfrecord(eval_files, batch_size, False)

    train_steps, val_steps = utils.get_steps(
        train_files, eval_files, batch_size)

    logdir = "src/logs/scalars/" + datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = keras.callbacks.TensorBoard(log_dir=logdir)

    early_stop = EarlyStopping(monitor="val_loss", patience=4)

    callbacks_list = [
        ModelCheckpoint('/home/alvaro/Desktop/multi-cue-sign-language/src/models/efficientnetV2_fine/', monitor='val_accuracy',
                        verbose=1, save_best_only=True, save_weights_only=True),
        LearningRateScheduler(lr_scheduler.lr_time_based_decay, verbose=1),
        tensorboard_callback,
        early_stop
    ]

    cnn_model = get_cnn_model(load_weights)
    recurrent_model = get_recurrent_model(learning_rate, cnn_model)

    if load_weights:
        recurrent_model.load_weights(
            '/home/alvaro/Desktop/multi-cue-sign-language/src/models/efficientnetV2/')

    recurrent_model.fit(train_gen(dataset),
                        steps_per_epoch=train_steps,
                        epochs=epochs,
                        validation_data=eval_gen(dataset_eval),
                        validation_steps=val_steps,
                        callbacks=callbacks_list)


if __name__ == '__main__':
    train_files = tf.io.gfile.glob(
        '/home/alvaro/Desktop/video2tfrecord/example/train_norm/*.tfrecords')

    eval_files = tf.io.gfile.glob(
        '/home/alvaro/Desktop/video2tfrecord/example/val_norm/*.tfrecords')

    epochs = 30
    batch_size = 8
    learning_rate = 0.00001
    train_cnn_lstm_model(train_files, eval_files, epochs,
                         batch_size, learning_rate, True)
