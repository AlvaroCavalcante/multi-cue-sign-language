import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt

from tensorflow import keras
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Concatenate, TimeDistributed
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler

from read_dataset import load_data_tfrecord
from utils import lr_scheduler
from utils import cnn_models

physical_devices = tf.config.list_physical_devices('GPU')
try:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
except:
    print('GPU not found')


MAX_SEQ_LENGTH = 16
NUMBER_OF_CLASSES = 20
HAND_WIDTH, HAND_HEIGHT = 80, 80
FACE_WIDTH, FACE_HEIGHT = 80, 80


def get_recurrent_model(learning_rate, cnn_model):
    frame_features_input = [keras.Input(
        (16, HAND_WIDTH, HAND_HEIGHT, 3), name="input"+str(c)) for c in range(3)]
    frame_features_input.append(keras.Input((16, 13), name='triangle_data'))

    # Refer to the following tutorial to understand the significance of using `mask`:
    # https://keras.io/api/layers/recurrent_layers/gru/
    x = TimeDistributed(cnn_model)(frame_features_input)
    x = keras.layers.GRU(16, return_sequences=True)(x)
    x = keras.layers.GRU(12)(x)
    x = keras.layers.Dropout(0.4)(x)
    x = keras.layers.Dense(8, activation="relu")(x)
    output = keras.layers.Dense(NUMBER_OF_CLASSES, activation="softmax")(x)

    rnn_model = keras.Model(frame_features_input, output)

    rnn_model.compile(
        loss="sparse_categorical_crossentropy", optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate), metrics=["accuracy"]
    )

    # tf.keras.utils.plot_model(rnn_model, "model.png", show_shapes=True)
    print(rnn_model.summary())
    return rnn_model


def get_hand_sequence(input_1, input_2):
    merged = Concatenate()([input_1, input_2])
    cnn_model = cnn_models.get_mobilenet_model(merged, prefix_name='hand')

    return cnn_model


def get_face_sequence(face_input):
    cnn_model = cnn_models.get_mobilenet_model(face_input, prefix_name='face')
    return cnn_model


def get_cnn_model():
    hand1_input = tf.keras.layers.Input(
        shape=(HAND_WIDTH, HAND_HEIGHT, 3), name='hand1_input')
    hand2_input = tf.keras.layers.Input(
        shape=(HAND_WIDTH, HAND_HEIGHT, 3), name='hand2_input')
    face_input = tf.keras.layers.Input(
        shape=(FACE_WIDTH, FACE_HEIGHT, 3), name='face_input')
    triangle_input = tf.keras.layers.Input(shape=(13), name='triangle_input')

    hand_seq = get_hand_sequence(hand1_input, hand2_input)
    face_seq = get_face_sequence(face_input)

    concat_layers = Concatenate()([hand_seq, face_seq])
    final_output = Concatenate()([concat_layers, triangle_input])

    model = Model(inputs=[hand1_input, hand2_input,
                  face_input, triangle_input], outputs=final_output)
    # tf.keras.utils.plot_model(model, "model.png", show_shapes=True)

    return model


def count_data_items(tfrecord):
    count = 0
    for fn in tfrecord:
        for _ in tf.compat.v1.python_io.tf_record_iterator(fn):
            count += 1

    return count

def plot_dist(label_arr):
    plt.bar(list(label_arr.keys()), height=list(label_arr.values()), color='blue')
    plt.savefig('label_dist.png')

label_dist = {}
plot_label_dist = False

def train_gen(dataset):
    for (hand_seq, face_seq, triangle_data, centroids, label, video_name_list, triangle_stream) in dataset:

        if plot_label_dist:
            for lbl in label:
                lbl = tf.constant(lbl).numpy()
                if not label_dist.get(lbl):
                    label_dist[int(lbl)] = 1
                else:
                    label_dist[int(lbl)] += 1

            plot_dist(label_dist)

        yield [hand_seq[:, 0], hand_seq[:, 1], face_seq, triangle_data], label


def train_cnn_lstm_model(train_files, epochs, batch_size, learning_rate):
    dataset = load_data_tfrecord(train_files, batch_size)

    num_training_imgs = count_data_items(train_files)
    print('Number of training images:', num_training_imgs)

    train_steps = num_training_imgs // batch_size
    print('Training steps: ', train_steps)

    callbacks_list = [
        ModelCheckpoint('src/model', monitor='accuracy',
                        verbose=1, save_best_only=True),
        LearningRateScheduler(lr_scheduler.lr_time_based_decay, verbose=1)
    ]

    cnn_model = get_cnn_model()
    recurrent_model = get_recurrent_model(learning_rate, cnn_model)

    result = recurrent_model.fit(train_gen(dataset),
                                      steps_per_epoch=train_steps,
                                      epochs=epochs,
                                      callbacks=callbacks_list)

    history_frame = pd.DataFrame(result.history)
    history_frame.to_csv('src/history.csv', index=False)
    return history_frame


if __name__ == '__main__':
    train_files =  tf.io.gfile.glob(
    '/home/alvaro/Documentos/video2tfrecord/example/train/*.tfrecords')
    epochs = 50
    batch_size = 6
    learning_rate = 0.001
    train_cnn_lstm_model(train_files, epochs, batch_size, learning_rate)
