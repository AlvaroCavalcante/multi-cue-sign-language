import tensorflow as tf

def get_image(img, width, height):
    image = tf.image.decode_jpeg(img, channels=3)
    image = tf.image.resize(image, [width, height])
    # image = tf.reshape(image, tf.stack([height, width, 3]))
    # image = tf.reshape(image, [1, height, width, 3])
    # image = tf.cast(image, dtype='uint8')
    image = tf.image.per_image_standardization(image)
    return image

def read_tfrecord(example_proto):
    face = []
    hand_1 = []
    hand_2 = []
    video_name = []
    triangle_stream_arr = []
    triangle_data = []
    centroids = []
    video = []

    for image_count in range(16):
        face_stream = 'face/' + str(image_count)
        hand_1_stream = 'hand_1/' + str(image_count)
        hand_2_stream = 'hand_2/' + str(image_count)
        video_stream = 'video/' + str(image_count)
        triangle_stream = 'triangle_data/' + str(image_count)
        centroid_stream = 'centroid/' + str(image_count)

        feature_dict = {
            face_stream: tf.io.FixedLenFeature([], tf.string),
            hand_1_stream: tf.io.FixedLenFeature([], tf.string),
            hand_2_stream: tf.io.FixedLenFeature([], tf.string),
            video_stream: tf.io.FixedLenFeature([], tf.string),
            triangle_stream: tf.io.VarLenFeature(tf.float32),
            centroid_stream: tf.io.VarLenFeature(tf.float32),
            'video_name': tf.io.FixedLenFeature([], tf.string),
            'height': tf.io.FixedLenFeature([], tf.int64),
            'width': tf.io.FixedLenFeature([], tf.int64),
            'depth': tf.io.FixedLenFeature([], tf.int64),
            'label': tf.io.FixedLenFeature([], tf.int64)
        }

        features = tf.io.parse_single_example(
            example_proto, features=feature_dict)

        triangle_data.append(tf.squeeze(tf.reshape(
            features[triangle_stream].values, (1, 13))))

        centroids.append(tf.reshape(features[centroid_stream].values, (3, 2)))

        triangle_stream_arr.append(triangle_stream)

        width = tf.cast(features['width'], tf.int32)
        height = tf.cast(features['height'], tf.int32)

        face_image = get_image(features[face_stream], width, height)
        hand_1_image = get_image(features[hand_1_stream], width, height)
        hand_2_image = get_image(features[hand_2_stream], width, height)
        image = get_image(features[video_stream], 512, 512)

        face.append(face_image)
        hand_1.append(hand_1_image)
        hand_2.append(hand_2_image)
        video.append(image)
        video_name.append(features['video_name'])
        label = tf.cast(features['label'], tf.int32)

    return [hand_1, hand_2], face, triangle_data, centroids, video, label, video_name, triangle_stream_arr

def load_dataset(tf_record_path):
    raw_dataset = tf.data.TFRecordDataset(tf_record_path)
    parsed_dataset = raw_dataset.map(read_tfrecord)
    return parsed_dataset

def prepare_for_training(ds, batch_size, shuffle_buffer_size=25):
    # ds.cache() # I can remove this to don't use cache or use cocodata.tfcache

    ds = ds.repeat().shuffle(buffer_size=shuffle_buffer_size).batch(
        batch_size).prefetch(tf.data.experimental.AUTOTUNE)

    return ds

def load_data_tfrecord(tfrecord_path, batch_size):
    dataset = load_dataset(tfrecord_path)

    dataset = prepare_for_training(dataset, batch_size)
    return dataset