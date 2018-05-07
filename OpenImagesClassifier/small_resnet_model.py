"""Small network model that uses residual learning"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

BATCH_NORMALIZATION_MOMENTUM = 0.997


def batch_norm(input, training):
    """Adds batch normalization to graph.
    Args:
        input: previous op/tensor
        training: Indicator for usage while training"""
    # epsilon use default
    return tf.layers.batch_normalization(input, training=training,
                                         momentum=BATCH_NORMALIZATION_MOMENTUM, fused=True,
                                         center=True, scale=True,)


def residual_unit(input, filters_num, kernel_size, training, name):
    """Builds one residual unit
    Args:
        input: previous op/tensor
        filters_num: Number of resulting filters / feature-maps
        kernel_size: size of convolution kernel
        training: Indicator for usage while training
        name: name for layer"""
    with tf.name_scope(name):
        shortcut = input
        first_layer_stride = 1
        # if number of filters is doubled the size of the image representation is halved, shortcut have to be projected
        if int(shortcut.get_shape()[-1]) != filters_num:
            transform_conv = tf.layers.conv2d(input, filters=filters_num, kernel_size=1, strides=2, padding='SAME')
            shortcut = batch_norm(transform_conv, training)
            first_layer_stride = 2

        conv_1 = tf.layers.conv2d(input, filters=filters_num, kernel_size=kernel_size, padding='SAME',
                                  strides=first_layer_stride, use_bias=False)
        bn_1 = batch_norm(conv_1, training)
        relu_1 = tf.nn.relu(bn_1)

        conv_2 = tf.layers.conv2d(relu_1, filters=filters_num, kernel_size=kernel_size, padding='SAME',
                                  use_bias=False)
        bn_2 = batch_norm(conv_2, training)
        residual = tf.add(shortcut, bn_2)

        relu_2 = tf.nn.relu(residual)
        return tf.identity(relu_2, name=name)


def build_small_resnet(input, classes_count, training):
    """Builds a small resnet model for predicting classes in training or productive mode
    Args:
        input: Tensor that represents input batch. Schema: [NHWC]
        classes_count: Number of classes that should be predicted through this model
        training: Boolean for training (True) or productive usage"""

    with tf.variable_scope('Small_ResNet'):
        # input size 224x224x3
        # output size 112x112x32
        first_conv = tf.layers.conv2d(input, filters=32, kernel_size=7, padding='SAME', strides=2, name="First_Conv")
        first_bn = batch_norm(first_conv, training)
        first_relu = tf.nn.relu(first_bn)

        # output size 56x56x32
        max_pool = tf.layers.max_pooling2d(first_relu, pool_size=3, strides=2, padding='SAME', name="Max_Pooling")

        # output size 56x56x32
        res_unit_1 = residual_unit(max_pool, filters_num=32, kernel_size=3, training=training, name="ResUnit_1")
        res_unit_2 = residual_unit(res_unit_1, filters_num=32, kernel_size=3, training=training, name="ResUnit_2")

        # output size 28x28x64
        res_unit_3 = residual_unit(res_unit_2, filters_num=64, kernel_size=3, training=training, name="ResUnit_3")
        res_unit_4 = residual_unit(res_unit_3, filters_num=64, kernel_size=3, training=training, name="ResUnit_4")

        # output size 14x14x128
        res_unit_5 = residual_unit(res_unit_4, filters_num=128, kernel_size=3, training=training, name="ResUnit_5")

        # output size 1x1x128
        avg_pool = tf.layers.average_pooling2d(res_unit_5, pool_size=14, padding='VALID', strides=1, name="Avg_Pooling")
        reshaped = tf.reshape(avg_pool, [-1, 128])
        logits = tf.layers.dense(inputs=reshaped, units=classes_count, name="Logits")

        return logits

