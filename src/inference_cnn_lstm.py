import time

import numpy as np
import pandas as pd
import tensorflow as tf

from read_dataset import load_data_tfrecord
import cnn_lstm
from utils import utils


def disable_gpu():
    tf.config.set_visible_devices([], 'GPU')
    visible_devices = tf.config.get_visible_devices()
    for device in visible_devices:
        assert device.device_type != 'GPU'


def eval_gen(dataset):
    for (data, label) in dataset:
        yield data, label


def run_model_inference(files, model_path, batch_size, evaluate=True, use_gpu=True, eval_model_speed=True):
    if not use_gpu:
        disable_gpu()

    n_data = utils.count_data_items(files)
    dataset = load_data_tfrecord(files, batch_size, False)

    model = get_model(model_path)

    if evaluate:
        accuracy = model.evaluate(eval_gen(dataset), steps=n_data//batch_size)
        print(accuracy)

    if eval_model_speed:
        evaluate_model_speed(model, dataset)
    else:
        run_inference_and_save_dataframe(dataset, model)


def run_inference_and_save_dataframe(dataset, model):
    class_vocab = pd.read_csv('./src/utils/class_id_correspondence.csv')
    predictions = []
    video_names = []
    correct_prediction = []

    for data, label, video_name in dataset:
        probabilities = model.predict(data)
        class_prediction = list([np.argmax(proba) for proba in probabilities])
        predictions.extend(class_prediction)
        video_names.extend(list([name.numpy() for name in video_name]))
        correct_prediction.extend(list(
            [label[i].numpy() in np.argsort(proba)[::-1][0:3] for i, proba in enumerate(probabilities)]))

    prediction_df = pd.DataFrame(
        {'predictions': predictions, 'video_names': video_names, 'correct_prediction': correct_prediction})
    prediction_df.to_csv('top3_predictions_lstm.csv', index=False)


def evaluate_model_speed(model, dataset):
    inference_times = []

    for data, label in dataset:
        start_inference_time = time.time()
        model.predict(data)
        inference_time = time.time() - start_inference_time
        inference_times.append(inference_time)

    print('Average inference time: ', np.mean(inference_times))


def get_model(model_path):
    cnn_model = cnn_lstm.get_cnn_model(False)
    model = cnn_lstm.get_recurrent_model(0.0001, cnn_model)
    model.load_weights(model_path).expect_partial()
    return model


if __name__ == '__main__':
    files = tf.io.gfile.glob(
        '/home/alvaro/Desktop/video2tfrecord/results/test_v5/*.tfrecords')

    model_path = '/home/alvaro/Desktop/multi-cue-sign-language/src/models/step1_hands_fine_v2/'

    run_model_inference(files, model_path, batch_size=1,
                        evaluate=False, use_gpu=False, eval_model_speed=True)
