import tensorflow as tf
import tensorflow.keras.backend as K
import numpy as np


def get_param_count(model):
    trainable_count = np.sum([K.count_params(w)
                              for w in model.trainable_weights])
    non_trainable_count = np.sum([K.count_params(w)
                                  for w in model.non_trainable_weights])
    print('Total params: {:,}'.format(trainable_count + non_trainable_count))
    print('Trainable params: {:,}'.format(trainable_count))
    print('Non-trainable params: {:,}'.format(non_trainable_count))


def count_data_items(tfrecord):
    count = 0
    for fn in tfrecord:
        for _ in tf.compat.v1.python_io.tf_record_iterator(fn):
            count += 1

    return count


def get_steps(train_files, eval_files, batch_size):
    num_training_videos = 27500 # count_data_items(train_files) # 28112 # 27500
    print('Number of training videos:', num_training_videos)

    num_val_videos = 4015 # count_data_items(eval_files) # 4030 # 4015
    print('Number of validation videos:', num_val_videos)

    train_steps = num_training_videos // batch_size
    print('Training steps: ', train_steps)

    val_steps = num_val_videos // batch_size
    print('Validation steps: ', val_steps)

    return train_steps, val_steps
