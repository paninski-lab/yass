import tensorflow as tf


def weight_variable(shape, varName=None):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial, name=varName)


def bias_variable(shape, varName=None):
    initial = tf.constant(0.0, shape=shape)
    return tf.Variable(initial, name=varName)


def conv2d(x, W):
    """
    Performs 2-dimensional convolution with SAME padding.

    Parameters:
    -----------
    x: tf.Variable
        input data.
    W: tf.Variable
        weight matrix to be convolved with x.

    Returns:
    -----------
    x_convolved: tf.Variable
        output of the convolution function with the same shape as x.
    """

    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def conv2d_VALID(x, W):
    """
    Performs 2-dimensional convolution with VALID padding.

    Parameters:
    -----------
    x: tf.Variable
        input data.
    W: tf.Variable
        weight matrix to be convolved with x.

    Returns:
    -----------
    x_convolved: tf.Variable
        output of the convolution function with smaller shape than x.
    """

    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='VALID')


def max_pool(x, W):
    return tf.nn.max_pool(x, W, strides=[1, 1, 1, 1], padding='SAME')
