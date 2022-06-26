import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
import tensorflow as tf

from read_dataset import load_data_tfrecord
import cnn_lstm

tf.config.set_visible_devices([], 'GPU')
visible_devices = tf.config.get_visible_devices()
for device in visible_devices:
    assert device.device_type != 'GPU'

files = tf.io.gfile.glob(
        '/home/alvaro/Desktop/video2tfrecord/example/train/*.tfrecords')

dataset = load_data_tfrecord(files, 5, False)

def eval_gen():
    for (hand_seq, face_seq, triangle_data, centroids, label, video_name_list, triangle_stream) in dataset:
        yield [hand_seq[:, 0], hand_seq[:, 1], face_seq, triangle_data], label


cnn_model = cnn_lstm.get_cnn_model(False)
recurrent_model = cnn_lstm.get_recurrent_model(0.0001, cnn_model)

recurrent_model.load_weights(
            '/home/alvaro/Desktop/multi-cue-sign-language/src/models/efficient_net_b0_fine_v2/')

class_vocab = pd.read_csv('./src/utils/class_id_correspondence.csv')

result = recurrent_model.evaluate(dataset)

for data, label in eval_gen():
    probabilities = recurrent_model.predict(data)[0]
    class_prediction = np.argsort(probabilities)[::-1]

    print(f"  {class_vocab.iloc[i]['EN']}: {probabilities[i] * 100:5.2f}%")

