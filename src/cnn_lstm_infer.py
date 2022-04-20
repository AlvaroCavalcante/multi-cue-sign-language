import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model

from read_dataset import load_data_tfrecord

dataset = load_data_tfrecord('./src/batch_1_of_98_2022-04-19-01-04-1650330291.tfrecords', 1)

def eval_gen():
    for (hand_seq, face_seq, triangle_data, centroids, label, video_name_list, triangle_stream) in dataset:
        yield [hand_seq[:, 0], hand_seq[:, 1], face_seq, triangle_data], video_name_list[0][0].numpy()

model = load_model('/home/alvaro/Documentos/model')
class_vocab = pd.read_csv('./src/class_id_correspondence.csv')

for data, video_name in eval_gen():
    probabilities = model.predict(data)[0]
    for i in np.argsort(probabilities)[::-1]:
        print(f"  {class_vocab.iloc[i]['EN']}: {probabilities[i] * 100:5.2f}%")
    
