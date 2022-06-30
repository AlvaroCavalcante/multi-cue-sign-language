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


def get_transform_matrix(apply_proba_dict, range_aug_dict, is_hand=False):
    one = tf.constant([1], dtype='float32')
    zero = tf.constant([0], dtype='float32')
    identity_matrix = tf.cast(tf.reshape(tf.concat(
        [one, zero, zero, zero, one, zero, zero, zero, one], axis=0), [3, 3]), dtype='float32')

    transform_matrix = identity_matrix

    if apply_proba_dict['rotation'] > 0.5:
        rotation_matrix = get_rotation_matrix(
            range_aug_dict['rotation'], zero, one)
        transform_matrix = multiply_matrix(
            transform_matrix, rotation_matrix, identity_matrix)

    if apply_proba_dict['shear'] > 0.5:
        shear_matrix = get_shear_matrix(range_aug_dict['shear'], zero, one)
        transform_matrix = multiply_matrix(
            transform_matrix, shear_matrix, identity_matrix)

    if apply_proba_dict['zoom'] > 0.5 and not is_hand:
        zoom_matrix = tf.reshape(tf.concat(
            [one/range_aug_dict['height_zoom'], zero, zero, zero, one/range_aug_dict['width_zoom'], zero, zero, zero, one], axis=0), [3, 3])
        transform_matrix = multiply_matrix(
            transform_matrix, zoom_matrix, identity_matrix)

    if apply_proba_dict['shift'] > 0.5 and not is_hand:
        shift_matrix = tf.reshape(tf.concat(
            [one, zero, range_aug_dict['height_shift'], zero, one, range_aug_dict['width_shift'], zero, zero, one], axis=0), [3, 3])
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


def transform_image(image, img_width, apply_proba_dict, range_aug_dict, seed, is_hand=False):
    DIM = img_width
    XDIM = DIM % 2

    if apply_proba_dict.get('brightness') > 0.5:
        image = tf.image.random_brightness(image, 0.20, seed=seed)
    if apply_proba_dict.get('contrast') > 0.5:
        image = tf.image.random_contrast(image, 0.7, 2, seed=seed)
    if apply_proba_dict.get('saturation') > 0.5:
        image = tf.image.random_saturation(image, 0.75, 1.25, seed=seed)
    if apply_proba_dict.get('hue') > 0.5:
        image = tf.image.random_hue(image, 0.1, seed=seed)
    if apply_proba_dict.get('flip_left_right') > 0.5:
        image = tf.image.random_flip_left_right(image, seed=seed)

    transform_matrix = get_transform_matrix(apply_proba_dict, range_aug_dict, is_hand)
    image = apply_operation(image, transform_matrix, DIM, XDIM)

    return image
