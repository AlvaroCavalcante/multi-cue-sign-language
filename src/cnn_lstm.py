import pickle

import tensorflow as tf
import pandas as pd
from tensorflow import keras
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Concatenate, Conv2D, MaxPooling2D, GlobalAveragePooling2D, TimeDistributed, Flatten
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler

from read_dataset import load_data_tfrecord
import lr_scheduler

physical_devices = tf.config.list_physical_devices('GPU')
try:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
except:
    print('GPU not found')

MAX_SEQ_LENGTH = 16
NUM_FEATURES = 3000
NUMBER_OF_CLASSES = 3
HAND_WIDTH, HAND_HEIGHT = 50, 50
FACE_WIDTH, FACE_HEIGHT = 50, 50

def get_sequence_model():
    cnn_model = get_cnn_model()

    frame_features_input = [keras.Input((16, HAND_WIDTH, HAND_HEIGHT, 3), name="input"+str(c)) for c in range(3)]
    frame_features_input.append(keras.Input((16, 13), name='triangle_data'))

    # Refer to the following tutorial to understand the significance of using `mask`:
    # https://keras.io/api/layers/recurrent_layers/gru/
    x = TimeDistributed(cnn_model)(frame_features_input)
    x = keras.layers.GRU(16, return_sequences=True)(x)
    x = keras.layers.GRU(8)(x)
    x = keras.layers.Dropout(0.4)(x)
    x = keras.layers.Dense(8, activation="relu")(x)
    output = keras.layers.Dense(NUMBER_OF_CLASSES, activation="softmax")(x)

    rnn_model = keras.Model(frame_features_input, output)

    rnn_model.compile(
        loss="sparse_categorical_crossentropy", optimizer=tf.keras.optimizers.Adam(learning_rate=0.01), metrics=["accuracy"]
    )

    # tf.keras.utils.plot_model(rnn_model, "model.png", show_shapes=True)
    return rnn_model


def get_base_sequence(input):
    x = Conv2D(filters=64, kernel_size=3, padding="same",activation="relu")(input)
    x = MaxPooling2D(pool_size=2)(x)
    x = Conv2D(filters=32, kernel_size=3, padding="same",activation="relu")(x)
    x = MaxPooling2D(pool_size=2)(x)
    x = Conv2D(filters=16, kernel_size=3, padding="same",activation="relu")(x)
    x = MaxPooling2D(pool_size=2)(x)
    output = GlobalAveragePooling2D()(x)
    return output

def get_cnn_model():
    hand1_input = tf.keras.layers.Input(shape=(HAND_WIDTH, HAND_HEIGHT, 3), name='hand1_input')
    hand2_input = tf.keras.layers.Input(shape=(HAND_WIDTH, HAND_HEIGHT, 3), name='hand2_input')
    face_input = tf.keras.layers.Input(shape=(FACE_WIDTH, FACE_HEIGHT, 3), name='face_input')
    triangle_input = tf.keras.layers.Input(shape=(13), name='triangle_input')

    hand1_seq = get_base_sequence(hand1_input)
    hand2_seq = get_base_sequence(hand2_input)
    face_seq = get_base_sequence(face_input)

    concat_layers = Concatenate()([hand1_seq, hand2_seq, face_seq])
    final_output = Concatenate()([concat_layers, triangle_input])

    model = Model(inputs=[hand1_input, hand2_input, face_input, triangle_input], outputs=final_output)
    # tf.keras.utils.plot_model(model, "model.png", show_shapes=True)

    return model

def count_data_items(tfrecord):
    count = 0
    for fn in tfrecord:
      for _ in tf.compat.v1.python_io.tf_record_iterator(fn):
        count += 1

    return count

train_files = tf.io.gfile.glob('./src/data/*.tfrecords')

batch_size = 6

dataset = load_data_tfrecord(train_files, batch_size)

num_training_imgs = count_data_items(train_files)

train_steps = num_training_imgs // batch_size

a_file = open("./src/data.pkl", "rb")
rep_dict = pickle.load(a_file)
rep_dict = dict((v, k) for k, v in rep_dict.items())

def train_gen():
    rep_dict = {104: 2, 1: 1, 118: 2 }
    for (hand_seq, face_seq, triangle_data, centroids, label, video_name_list, triangle_stream) in dataset:
        yield [hand_seq[:, 0], hand_seq[:, 1], face_seq, triangle_data], tf.constant([rep_dict[x.numpy()] for x in label], dtype=tf.int16)

callbacks_list = [
    ModelCheckpoint('src/model', monitor='accuracy', verbose=1, save_best_only=True),
    LearningRateScheduler(lr_scheduler.lr_time_based_decay, verbose=1)
]

result = get_sequence_model().fit(train_gen(),
                    steps_per_epoch=train_steps,
                    epochs=80,
                    callbacks=callbacks_list)

history_frame = pd.DataFrame(result.history)
history_frame.to_csv('src/history.csv', index=False)