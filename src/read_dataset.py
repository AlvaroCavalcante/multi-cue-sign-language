import random

import tensorflow as tf
from data_augmentation import transform_image
# from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
# from tensorflow.keras.applications.xception import preprocess_input


def get_image(img, width, height):
    image = tf.image.decode_jpeg(img, channels=3)
    image = tf.image.resize(image, [width, height])
    # image = tf.reshape(image, tf.stack([height, width, 3]))
    # image = tf.reshape(image, [1, height, width, 3])
    # image = tf.cast(image, dtype='uint8')
    # image = tf.image.per_image_standardization(image)
    # image = preprocess_input(image)
    return image


def get_apply_proba_dict():
    apply_proba_dict = {}
    aug_keys = ['brightness', 'contrast', 'saturation', 'hue',
                'flip_left_right', 'rotation', 'shear', 'zoom', 'shift']

    apply_proba_dict = {}

    for key in aug_keys:
        apply_proba_dict[key] = tf.random.uniform([], 0, 1.0, dtype=tf.float32)

    return apply_proba_dict


def get_range_aug_dict(img_width):
    range_aug_dict = {}

    rotation_range = [-20, 20]
    shear_range = [5, 12]
    h_zoom_range = [0.8, 1.2]
    w_zoom_range = [0.8, 1.2]
    h_shift_range = [0, 0.15]
    w_shift_range = [0, 0.05]

    range_aug_dict['rotation'] = tf.random.uniform([1], rotation_range[0],
                                                   rotation_range[1], dtype=tf.float32)
    range_aug_dict['shear'] = tf.random.uniform([1], shear_range[0],
                                                shear_range[1], dtype=tf.float32)
    range_aug_dict['height_zoom'] = tf.random.uniform(
        [1], h_zoom_range[0], h_zoom_range[1], dtype=tf.float32)
    range_aug_dict['width_zoom'] = tf.random.uniform(
        [1], w_zoom_range[0], w_zoom_range[1], dtype=tf.float32)
    range_aug_dict['height_shift'] = tf.random.uniform(
        [1], h_shift_range[0], h_shift_range[1], dtype=tf.float32) * img_width
    range_aug_dict['width_shift'] = tf.random.uniform(
        [1], w_shift_range[0], w_shift_range[1], dtype=tf.float32) * img_width

    return range_aug_dict


def read_tfrecord_test(example_proto):
    face = []
    hand_1 = []
    hand_2 = []
    triangle_data = []
    face_keypoints = []

    for image_count in range(16):
        face_stream = 'face/' + str(image_count)
        hand_1_stream = 'hand_1/' + str(image_count)
        hand_2_stream = 'hand_2/' + str(image_count)
        triangle_stream = 'triangle_data/' + str(image_count)
        moviment_stream = 'moviment/' + str(image_count)
        keypoint_stream = 'keypoint/' + str(image_count)

        feature_dict = {
            face_stream: tf.io.FixedLenFeature([], tf.string),
            hand_1_stream: tf.io.FixedLenFeature([], tf.string),
            hand_2_stream: tf.io.FixedLenFeature([], tf.string),
            triangle_stream: tf.io.VarLenFeature(tf.float32),
            moviment_stream: tf.io.VarLenFeature(tf.float32),
            keypoint_stream: tf.io.VarLenFeature(tf.float32),
            'video_name': tf.io.FixedLenFeature([], tf.string),
            'height': tf.io.FixedLenFeature([], tf.int64),
            'width': tf.io.FixedLenFeature([], tf.int64),
            'label': tf.io.FixedLenFeature([], tf.int64)
        }

        features = tf.io.parse_single_example(
            example_proto, features=feature_dict)

        face_keypoints.append(tf.squeeze(tf.reshape(
            features[keypoint_stream].values, (1, 136))))

        triangle = tf.squeeze(tf.reshape(
            features[triangle_stream].values, (1, 11)))

        moviment = tf.squeeze(
            tf.reshape(features[moviment_stream].values, (1, 2)))

        triangle_data.append(tf.concat([triangle, moviment], axis=0))

        width = tf.cast(features['width'], tf.int32)
        height = tf.cast(features['height'], tf.int32)

        face_image = get_image(features[face_stream], width, height)
        hand_1_image = get_image(features[hand_1_stream], width, height)
        hand_2_image = get_image(features[hand_2_stream], width, height)

        face.append(face_image)
        hand_1.append(hand_1_image)
        hand_2.append(hand_2_image)

    label = tf.cast(features['label'], tf.int32)

    return (hand_1, hand_2, triangle_data, face_keypoints), label


def read_tfrecord_train(example_proto):
    face = []
    hand_1 = []
    hand_2 = []
    triangle_data = []
    face_keypoints = []

    apply_proba_dict = get_apply_proba_dict()
    range_aug_dict = get_range_aug_dict(80)
    seed = random.randint(0, 10000)

    for image_count in range(16):
        face_stream = 'face/' + str(image_count)
        hand_1_stream = 'hand_1/' + str(image_count)
        hand_2_stream = 'hand_2/' + str(image_count)
        triangle_stream = 'triangle_data/' + str(image_count)
        moviment_stream = 'moviment/' + str(image_count)
        keypoint_stream = 'keypoint/' + str(image_count)

        feature_dict = {
            face_stream: tf.io.FixedLenFeature([], tf.string),
            hand_1_stream: tf.io.FixedLenFeature([], tf.string),
            hand_2_stream: tf.io.FixedLenFeature([], tf.string),
            triangle_stream: tf.io.VarLenFeature(tf.float32),
            moviment_stream: tf.io.VarLenFeature(tf.float32),
            keypoint_stream: tf.io.VarLenFeature(tf.float32),
            'video_name': tf.io.FixedLenFeature([], tf.string),
            'height': tf.io.FixedLenFeature([], tf.int64),
            'width': tf.io.FixedLenFeature([], tf.int64),
            'depth': tf.io.FixedLenFeature([], tf.int64),
            'label': tf.io.FixedLenFeature([], tf.int64)
        }

        features = tf.io.parse_single_example(
            example_proto, features=feature_dict)

        face_keypoints.append(tf.squeeze(tf.reshape(
            features[keypoint_stream].values, (1, 136))))

        triangle = tf.squeeze(tf.reshape(
            features[triangle_stream].values, (1, 11)))

        moviment = tf.squeeze(
            tf.reshape(features[moviment_stream].values, (1, 2)))

        triangle_data.append(tf.concat([triangle, moviment], axis=0))

        width = tf.cast(features['width'], tf.int32)
        height = tf.cast(features['height'], tf.int32)

        face_image = get_image(features[face_stream], width, height)
        hand_1_image = get_image(features[hand_1_stream], width, height)
        hand_2_image = get_image(features[hand_2_stream], width, height)

        face_image = transform_image(
            face_image, width, apply_proba_dict, range_aug_dict, seed)
        hand_1_image = transform_image(
            hand_1_image, width, apply_proba_dict, range_aug_dict, seed, True)
        hand_2_image = transform_image(
            hand_2_image, width, apply_proba_dict, range_aug_dict, seed, True)

        face.append(face_image)
        hand_1.append(hand_1_image)
        hand_2.append(hand_2_image)
        label = tf.cast(features['label'], tf.int32)

    return hand_1, hand_2, triangle_data, face_keypoints, label


def filter_func(hands, face, triangle_data, centroids, label, video_name, triangle_stream_arr):
    return tf.math.less(label, 20)


def load_dataset(tf_record_path, train):
    raw_dataset = tf.data.TFRecordDataset(tf_record_path)
    if train:
        parsed_dataset = raw_dataset.map(read_tfrecord_train)
    else:
        parsed_dataset = raw_dataset.map(read_tfrecord_test)

    return parsed_dataset


def prepare_data(ds, batch_size, train, shuffle_buffer_size=700):
    # ds.cache() # I can remove this to don't use cache or use cocodata.tfcache

    if train:
        ds = ds.repeat().shuffle(buffer_size=shuffle_buffer_size).batch(
            batch_size).prefetch(tf.data.AUTOTUNE)
    else:
        ds = ds.repeat().batch(
            batch_size)

    return ds


def load_data_tfrecord(tfrecord_path, batch_size, train=True):
    dataset = load_dataset(tfrecord_path, train)
    # dataset = dataset.filter(filter_func) # use this to get a small amount of classes

    dataset = prepare_data(dataset, batch_size, train)
    return dataset
