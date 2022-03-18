import tensorflow as tf
import pdb

FLAGS = tf.app.flags.FLAGS


def gyro_dropout(x, mask):
    input_shape = x.get_shape().as_list()
    mask_shape = mask.get_shape().as_list()
    num_blocks = [FLAGS.batch_size // mask_shape[0], input_shape[1] // mask_shape[1]]

    keep_prob = tf.count_nonzero(mask) / (mask_shape[0] * mask_shape[1])
    scale_value = tf.cast((1 / keep_prob), tf.float32)

    mask = tf.keras.backend.repeat_elements(mask, num_blocks[0], axis=0)

    return x * mask * scale_value

