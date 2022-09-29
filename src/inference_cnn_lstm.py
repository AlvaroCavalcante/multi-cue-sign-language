import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
import tensorflow as tf
from sklearn.metrics import confusion_matrix

from read_dataset import load_data_tfrecord
import cnn_lstm
from utils import utils

if False:  # disable GPU if necessary
    tf.config.set_visible_devices([], 'GPU')
    visible_devices = tf.config.get_visible_devices()
    for device in visible_devices:
        assert device.device_type != 'GPU'

files = tf.io.gfile.glob(
    '/home/alvaro/Desktop/video2tfrecord/results/test_v5/*.tfrecords')

n_data = utils.count_data_items(files)

dataset = load_data_tfrecord(files, 20, False)


def eval_gen(dataset):
    for (data, label) in dataset:
        yield data, label


def show_confusion_matrix(y_true, pred):
    confusion_mtx = confusion_matrix(y_true, pred)
    matrix_img = sns.heatmap(confusion_mtx, annot=True)
    fig = matrix_img.get_figure()
    fig.savefig("out.png")
    plt.show()


cnn_model = cnn_lstm.get_cnn_model(False)
recurrent_model = cnn_lstm.get_recurrent_model(0.0001, cnn_model)

recurrent_model.load_weights(
    '/home/alvaro/Desktop/multi-cue-sign-language/src/models/final_training_fine_v2/').expect_partial()

class_vocab = pd.read_csv('./src/utils/class_id_correspondence.csv')

result = recurrent_model.evaluate(eval_gen(dataset), steps=n_data//20) # number of data on val dataset by the batch size
print(result)

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

# true_categories = tf.concat([y for _, y, _ in dataset], axis=0)

# predictions = recurrent_model.predict(dataset)
# class_prediction = tf.argmax(predictions, axis=1)


# show_confusion_matrix(true_categories[0:500], class_prediction)
