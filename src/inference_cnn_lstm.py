import time

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Concatenate, TimeDistributed
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler, EarlyStopping
from tensorflow.keras.layers import Dropout, Dense
from tqdm import tqdm

from read_dataset import load_data_tfrecord
import cnn_lstm
from utils import rnn_models
from utils import utils


MAX_SEQ_LENGTH = 16
NUMBER_OF_CLASSES = 500
HAND_WIDTH, HAND_HEIGHT = 100, 200
FACE_WIDTH, FACE_HEIGHT = 100, 100
TRIANGLE_FIG_WIDTH, TRIANGLE_FIG_HEIGHT = 128, 128


def disable_gpu():
    tf.config.set_visible_devices([], 'GPU')
    visible_devices = tf.config.get_visible_devices()
    for device in visible_devices:
        assert device.device_type != 'GPU'
    print('GPU disabled')


def eval_gen(dataset):
    for (data, label) in dataset:
        yield data, label


def run_model_inference(files, model_path, batch_size, evaluate=True, use_gpu=True, eval_model_speed=True):
    if not use_gpu:
        disable_gpu()

    n_data = 4000  # utils.count_data_items(files)  # 33687
    dataset = load_data_tfrecord(files, batch_size, False)

    model = get_model(model_path)

    if evaluate:
        accuracy = model.evaluate(eval_gen(dataset), steps=n_data//batch_size)
        print(accuracy)

    if eval_model_speed:
        evaluate_model_speed(model, dataset, n_data)
    else:
        run_inference_and_save_dataframe(dataset, model)


def run_inference_and_save_dataframe(dataset, model):
    # class_vocab = pd.read_csv('./src/utils/class_id_correspondence.csv')
    predictions = []
    video_names = []
    correct_prediction = []
    labels = []

    for data, label, video_name in dataset:
        probabilities = model.predict(data)
        class_prediction = list([np.argmax(proba) for proba in probabilities])
        predictions.extend(class_prediction)
        video_names.extend(list([name.numpy() for name in video_name]))
        correct_prediction.extend(list(
            [label[i].numpy() == np.argsort(proba)[::-1][0] for i, proba in enumerate(probabilities)]))
        labels.append(label.numpy()[0])

    prediction_df = pd.DataFrame(
        {'predictions': predictions, 'video_names': video_names, 'correct_prediction': correct_prediction, 'labels': labels})
    prediction_df.to_csv('predictions_csl.csv', index=False)


def evaluate_model_speed(model, dataset, n_data):
    inference_times = []

    with tqdm(total=n_data) as pbar:
        for data, _ in dataset:
            start_inference_time = time.time()
            model.predict(data)
            final_inference_time = time.time() - start_inference_time
            inference_times.append(final_inference_time)

            if len(inference_times) == n_data:
                break
            pbar.update(1)

    inference_times.pop(0)
    print('Average inference time: ', np.mean(inference_times))
    print('Inference std: ', np.std(inference_times))


def get_model(model_path):
    face_cnn = cnn_lstm.get_cnn_model(FACE_WIDTH, FACE_HEIGHT, 'face')
    face_model = rnn_models.get_face_rnn_model(
        face_cnn, 1e-3)

    hands_cnn = cnn_lstm.get_cnn_model(HAND_WIDTH, HAND_HEIGHT, 'hands')
    hands_model = rnn_models.get_hands_rnn_model(hands_cnn, 1e-3)

    # triangle_model = rnn_models.get_triangle_rnn_model(1e-3)

    # triangle_cnn = cnn_lstm.get_cnn_model(
    #     TRIANGLE_FIG_WIDTH, TRIANGLE_FIG_HEIGHT, 'triangle', model_name='mobilenet')
    # triangle_fig_model = rnn_models.get_triangle_figure_rnn_model(
    #     triangle_cnn, 1e-3)

    # triangle_fig_model.load_weights(
    #     '/home/alvaro/Desktop/multi-cue-sign-language/src/models/triangle_figure_csl/').expect_partial()

    # hands_model.load_weights(
    #     '/home/alvaro/Desktop/multi-cue-sign-language/src/models/step1_hands_csl_fine_v2/').expect_partial()

    # face_model.load_weights(
    #     '/home/alvaro/Desktop/multi-cue-sign-language/src/models/step1_face_csl_fine_v2/').expect_partial()

    concat_layers = Concatenate()([
        face_model.layers[-2].output, hands_model.layers[-2].output])

    output = Dense(500, activation='softmax')(concat_layers)

    rnn_model = keras.Model(
        [face_model.input, hands_model.input], output)

    rnn_model.compile(
        loss='sparse_categorical_crossentropy', optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3), metrics=['accuracy']
    )

    rnn_model.load_weights(
        '/home/alvaro/Desktop/multi-cue-sign-language/src/models/step2_hands_face_csl_v2/').expect_partial()

    print(rnn_model.summary())
    return rnn_model


if __name__ == '__main__':
    files = tf.io.gfile.glob(
        '/home/alvaro/Desktop/video2tfrecord/results/test_v2/*.tfrecords')

    model_path = '/home/alvaro/Desktop/multi-cue-sign-language/src/models/triangle_figure_face_hands_csl/'

    run_model_inference(files, model_path, batch_size=1,
                        evaluate=False, use_gpu=False, eval_model_speed=True)
