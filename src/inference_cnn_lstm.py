import time

import numpy as np
import pandas as pd
import tensorflow as tf
from tqdm import tqdm

from read_dataset import load_data_tfrecord
import cnn_lstm
from utils import utils


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

    n_data = utils.count_data_items(files)  # 3735
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
    cnn_model = cnn_lstm.get_cnn_model(128, 128, 'triangle_fig', False)
    model = cnn_lstm.get_recurrent_model(cnn_model, 1e-3)

    model.load_weights(model_path).expect_partial()
    model.summary()
    return model


if __name__ == '__main__':
    files = tf.io.gfile.glob(
        '/home/alvaro/Desktop/video2tfrecord/results/test_v6/*.tfrecords')

    model_path = '/home/alvaro/Desktop/multi-cue-sign-language/src/models/triangle_figure_face_hands/'

    run_model_inference(files, model_path, batch_size=1,
                        evaluate=False, use_gpu=False, eval_model_speed=True)
