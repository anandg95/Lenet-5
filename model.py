# create the lenet-5 model here
# refer https://engmrk.com/lenet-5-a-classic-cnn-architecture/ for architecture

import tensorflow as tf


def init_weight(shape):
    w = tf.random.truncated_normal(
        shape=shape, mean=0.0, stddev=1.0, dtype=tf.dtypes.float32
    )
    return tf.Variable(w)


def init_bias(shape):
    b = tf.zeros(shape=shape, dtype=tf.dtypes.float32)
    return tf.Variable(b)


def LeNet(x):
    # layer1 - convoutional
    # input 32x32x1 (x) - output 28x28x6
    # filter shape - 5x5x1-s-1
    # number of filters - 6

    conv1_W = init_weight((5, 5, 1, 6))
    conv1_b = init_bias(6)

    conv1 = (
        tf.nn.conv2d(x, filter=conv1_W, strides=[1, 1, 1, 1], padding="VALID") + conv1_b
    )
    conv1 = tf.nn.relu(conv1)

    # layer2 - average pooling
    # shape - 2x2 - s-2
    conv1 = tf.nn.avg_pool(
        conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="VALID"
    )

    # layer3 - convolutional
    # input 14x14x6 - output 10x10x16
    # filter shape - 5x5x6 - s-1
    # number of filters - 16
    # each channel in input is not connected to each channel in output
    # in fact, each input only contributes to 10 out of 16 outputs, and not all 16

    conv2_W = init_weight((5, 5, 6, 16))
    conv2_b = init_bias(16)
    conv2 = (
        tf.nn.conv2d(conv1, filter=conv2_W, strides=[1, 1, 1, 1], padding="VALID")
        + conv2_b
    )
    conv2 = tf.nn.relu(conv2)

    # layer 4 - average pooling
    # shape - 2x2 -s-2
    conv2 = tf.nn.avg_pool(
        conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="VALID"
    )

    # flatten out so as to pass onto FC layer
    # shape becomes 5x5x16 = (None, 400)
    fc0 = tf.contrib.layers.flatten(conv2)

    # Layer 5 - FC layer 1
    # input 400 - output 120
    fc1_W = init_weight((400, 120))
    fc1_b = init_bias(120)

    fc1 = tf.matmul(fc0, fc1_W) + fc1_b
    fc1 = tf.nn.relu(fc1)

    # Layer 6 - FC layer 2
    # input 400 - output 84
    fc2_W = init_weight((120, 84))
    fc2_b = init_bias(84)

    fc2 = tf.matmul(fc1, fc2_W) + fc2_b
    fc2 = tf.nn.relu(fc2)

    # Layer 7 - last layer, FC
    # input 84 - output 10 (num of classes)
    fc3_W = init_weight((84, 10))
    fc3_b = init_bias(10)

    logits = tf.matmul(fc2, fc3_W) + fc3_b

    return logits
