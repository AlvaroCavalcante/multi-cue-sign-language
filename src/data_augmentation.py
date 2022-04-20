import math

import tensorflow as tf
from tensorflow.keras import backend as K


def get_rotation_matrix(rotation, zero, one):
    rotation = math.pi * rotation / 180.
    c1 = tf.math.cos(rotation)
    s1 = tf.math.sin(rotation)
    return tf.reshape(tf.concat([c1, s1, zero, -s1, c1, zero, zero, zero, one], axis=0), [3, 3])


def get_shear_matrix(shear, zero, one):
    shear = math.pi * shear / 180.
    c2 = tf.math.cos(shear)
    s2 = tf.math.sin(shear)
    return tf.reshape(tf.concat([one, s2, zero, zero, c2, zero, zero, zero, one], axis=0), [3, 3])


def multiply_matrix(transformation, origin_matrix, identity_matrix):
    if tf.reduce_all(transformation == identity_matrix):
        return origin_matrix

    return K.dot(transformation, origin_matrix)


def get_transform_matrix(rotation, shear, height_zoom, width_zoom, height_shift, width_shift, is_hand=False):
    one = tf.constant([1], dtype='float32')
    zero = tf.constant([0], dtype='float32')
    identity_matrix = tf.cast(tf.reshape(tf.concat(
        [one, zero, zero, zero, one, zero, zero, zero, one], axis=0), [3, 3]), dtype='float32')

    transform_matrix = identity_matrix

    if tf.random.uniform([], 0, 1.0, dtype=tf.float32) > 0.5:
        rotation_matrix = get_rotation_matrix(rotation, zero, one)
        transform_matrix = multiply_matrix(
            transform_matrix, rotation_matrix, identity_matrix)
    else:
        rotation_matrix = identity_matrix

    if tf.random.uniform([], 0, 1.0, dtype=tf.float32) > 0.5:
        shear_matrix = get_shear_matrix(shear, zero, one)
        transform_matrix = multiply_matrix(
            transform_matrix, shear_matrix, identity_matrix)
    else:
        shear_matrix = identity_matrix

    if is_hand:
        return transform_matrix

    if tf.random.uniform([], 0, 1.0, dtype=tf.float32) > 0.5:
        zoom_matrix = tf.reshape(tf.concat(
            [one/height_zoom, zero, zero, zero, one/width_zoom, zero, zero, zero, one], axis=0), [3, 3])
        transform_matrix = multiply_matrix(
            transform_matrix, zoom_matrix, identity_matrix)
    else:
        zoom_matrix = identity_matrix

    if tf.random.uniform([], 0, 1.0, dtype=tf.float32) > 0.5:
        shift_matrix = tf.reshape(tf.concat(
            [one, zero, height_shift, zero, one, width_shift, zero, zero, one], axis=0), [3, 3])
    else:
        shift_matrix = identity_matrix
        transform_matrix = multiply_matrix(
            transform_matrix, shift_matrix, identity_matrix)

    return transform_matrix


def apply_operation(image, transform_matrix, DIM, XDIM):
    # LIST DESTINATION PIXEL INDICES
    x = tf.repeat(tf.range(DIM//2, -DIM//2, -1), DIM)
    y = tf.tile(tf.range(-DIM//2, DIM//2), [DIM])
    z = tf.ones([DIM*DIM], dtype='int32')
    idx = tf.stack([x, y, z])

    # ROTATE DESTINATION PIXELS ONTO ORIGIN PIXELS
    idx2 = K.dot(transform_matrix, tf.cast(idx, dtype='float32'))
    idx2 = K.cast(idx2, dtype='int32')
    idx2 = K.clip(idx2, -DIM//2+XDIM+1, DIM//2)

    # FIND ORIGIN PIXEL VALUES
    idx3 = tf.stack([DIM//2-idx2[0, ], DIM//2-1+idx2[1, ]])
    d = tf.gather_nd(image, tf.transpose(idx3))
    return tf.reshape(d, [DIM, DIM, 3])


def transform(image, img_width, is_hand=False):
    DIM = img_width
    XDIM = DIM % 2

    rotation_range = [-30, 30]
    shear_range = [1, 10]
    h_zoom_range = [0.8, 1.2]
    w_zoom_range = [0.8, 1.2]
    h_shift_range = [0, 0.15]
    w_shift_range = [0, 0.05]

    rot = tf.random.uniform([1], rotation_range[0],
                            rotation_range[1], dtype=tf.float32)
    shr = tf.random.uniform([1], shear_range[0],
                            shear_range[1], dtype=tf.float32)
    h_zoom = tf.random.uniform(
        [1], h_zoom_range[0], h_zoom_range[1], dtype=tf.float32)
    w_zoom = tf.random.uniform(
        [1], w_zoom_range[0], w_zoom_range[1], dtype=tf.float32)
    h_shift = tf.random.uniform(
        [1], h_shift_range[0], h_shift_range[1], dtype=tf.float32) * DIM
    w_shift = tf.random.uniform(
        [1], w_shift_range[0], w_shift_range[1], dtype=tf.float32) * DIM

    transform_matrix = get_transform_matrix(
        rot, shr, h_zoom, w_zoom, h_shift, w_shift, is_hand)
    transformed_image = apply_operation(image, transform_matrix, DIM, XDIM)

    return transformed_image


def transform_batch(face, hand_1, hand_2, img_width):
    face = transform(face, img_width)
    hand_img_1 = transform(hand_1, img_width, True)
    hand_img_2 = transform(hand_2, img_width, True)

    return face, hand_img_1, hand_img_2
