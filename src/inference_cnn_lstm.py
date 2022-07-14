import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
import tensorflow as tf
from sklearn.metrics import confusion_matrix

from read_dataset import load_data_tfrecord
import cnn_lstm

if False:  # disable GPU if necessary
    tf.config.set_visible_devices([], 'GPU')
    visible_devices = tf.config.get_visible_devices()
    for device in visible_devices:
        assert device.device_type != 'GPU'

files = tf.io.gfile.glob(
    '/home/alvaro/Desktop/video2tfrecord/example/train/*.tfrecords')

dataset = load_data_tfrecord(files, 24, False)


def eval_gen():
    for (hand_seq, face_seq, triangle_data, centroids, label, video_name_list, triangle_stream) in dataset:
        yield [hand_seq[:, 0], hand_seq[:, 1], face_seq, triangle_data], label


def show_confusion_matrix(y_true, pred):
    confusion_mtx = confusion_matrix(y_true, pred)
    matrix_img = sns.heatmap(confusion_mtx, annot=True)
    fig = matrix_img.get_figure()
    fig.savefig("out.png")
    plt.show()


cnn_model = cnn_lstm.get_cnn_model(False)
recurrent_model = cnn_lstm.get_recurrent_model(0.0001, cnn_model)

recurrent_model.load_weights(
    '/home/alvaro/Desktop/multi-cue-sign-language/src/models/lstm_efficientnet_fine/')

class_vocab = pd.read_csv('./src/utils/class_id_correspondence.csv')

# result = recurrent_model.evaluate(dataset)
# true_categories = tf.concat([y for _, y, _ in dataset], axis=0)

predictions = []
video_names = []
correct_prediction = []
count = 0

for data, label, video_name in dataset:
    probabilities = recurrent_model.predict(data)
    class_prediction = list([np.argmax(proba) for proba in probabilities])
    predictions.extend(class_prediction)
    video_names.extend(list([name.numpy() for name in video_name]))
    correct_prediction.extend(list(
        [label[i].numpy() in np.argsort(proba)[::-1][0:3] for i, proba in enumerate(probabilities)]))

prediction_df = pd.DataFrame(
    {'predictions': predictions, 'video_names': video_names, 'correct_prediction': correct_prediction})
prediction_df.to_csv('top3_predictions_lstm.csv', index=False)

# predictions = recurrent_model.predict(dataset)
# class_prediction = tf.argmax(predictions, axis=1)


# show_confusion_matrix(true_categories[0:500], class_prediction)
