# -*- coding: utf-8 -*-
# @File : MobileNet_v1.py
# @Author: Runist
# @Time : 2020/6/5 15:36
# @Software: PyCharm
# @Brief: MobileNet v1的实现

import os
import urllib.request
from tensorflow.keras import layers, models
from core.utils import process_bar


def depthwise_separable_convolution(inputs, depth_multiplier, pointwise_conv_filters, alpha, strides=(1, 1), block_id=1):
    """
    深度可分离卷积，depthwise_conv2d是一个卷积核负责一个通道，使得输出与输入通道数一致
    然后再用普通的1x1卷积核进行卷积
    :param inputs:
    :param depth_multiplier: 用来控制输出生成多少个通道的倍数
    :param pointwise_conv_filters: pw卷积的输出特征层的通道数
    :param alpha: 减少网络宽度的系数，0 < alpha < 1
    :param strides: 每个shape上的步长
    :return: 输出的channel是原来的2倍
    """
    if strides == (1, 1):
        x = inputs
    else:
        x = layers.ZeroPadding2D(((0, 1), (0, 1)), name='conv_pad_{}'.format(block_id))(inputs)

    # dw卷积采用DepthwiseConv2D，理论上只需要和输入层通道数一样就可以了，所以depth_multiplier为1
    # pw卷积来控制输出通道的数目
    x = layers.DepthwiseConv2D((3, 3), padding='same' if strides == (1, 1) else 'valid',
                               depth_multiplier=depth_multiplier,
                               strides=strides,
                               use_bias=False,
                               name='conv_dw_{}'.format(block_id))(x)

    x = layers.BatchNormalization(name='conv_dw_{}_bn'.format(block_id))(x)
    x = layers.ReLU(6., name='conv_dw_{}_relu'.format(block_id))(x)

    # pw卷积是常规1x1卷积，用来控制输出通道的数量
    pointwise_conv_filters = int(pointwise_conv_filters * alpha)
    x = layers.Conv2D(pointwise_conv_filters,
                      kernel_size=(1, 1),
                      strides=(1, 1),
                      padding="SAME",
                      use_bias=False,
                      name='conv_pw_{}'.format(block_id))(x)

    x = layers.BatchNormalization(name='conv_pw_{}_bn'.format(block_id))(x)
    x = layers.ReLU(6., name='conv_pw_{}_relu'.format(block_id))(x)

    return x


def conv_block(inputs, filters, kernel_size, alpha, strides):
    """
    ZeroPadding + Conv2D + BatchNormalization + relu
    :param inputs: 卷积块输入
    :param filters: 卷积核个数
    :param kernel_size: 卷积核大小
    :param alpha: 减少网络宽度的系数，0 < alpha < 1
    :param strides: 步长
    :return:
    """
    filters = int(filters * alpha)
    
    x = layers.ZeroPadding2D(padding=((0, 1), (0, 1)), name='conv1_pad')(inputs)
    x = layers.Conv2D(filters, kernel_size, strides=strides, use_bias=False, name='conv1')(x)
    x = layers.BatchNormalization(name='conv1_bn')(x)
    x = layers.ReLU(6., name='conv1_relu')(x)

    return x


def MobileNetV1(input_shape, num_classes, alpha, include_top=True):
    """
    MobileNetV1结构
    :param input_shape: 网络输入shape
    :param num_classes: 分类数量
    :param alpha: 减少网络宽度的系数，0 < alpha < 1
    :param include_top: 是否包含分类层
    :return: model
    """
    inputs = layers.Input(shape=input_shape, dtype='float32')

    x = conv_block(inputs, 32, 3, alpha, strides=2)
    x = depthwise_separable_convolution(x, 1, 64, alpha, block_id=1)
    x = depthwise_separable_convolution(x, 1, 128, alpha, block_id=2, strides=(2, 2))
    x = depthwise_separable_convolution(x, 1, 128, alpha, block_id=3)
    x = depthwise_separable_convolution(x, 1, 256, alpha, block_id=4, strides=(2, 2))
    x = depthwise_separable_convolution(x, 1, 256, alpha, block_id=5)
    x = depthwise_separable_convolution(x, 1, 512, alpha, block_id=6, strides=(2, 2))

    x = depthwise_separable_convolution(x, 1, 512, alpha, block_id=7)
    x = depthwise_separable_convolution(x, 1, 512, alpha, block_id=8)
    x = depthwise_separable_convolution(x, 1, 512, alpha, block_id=9)
    x = depthwise_separable_convolution(x, 1, 512, alpha, block_id=10)
    x = depthwise_separable_convolution(x, 1, 512, alpha, block_id=11)

    x = depthwise_separable_convolution(x, 1, 1024, alpha, block_id=12, strides=(2, 2))
    x = depthwise_separable_convolution(x, 1, 1024, alpha, block_id=13)

    x = layers.GlobalAveragePooling2D()(x)

    if include_top:
        outputs = layers.Dense(num_classes, name="prediction", activation="softmax")(x)
    else:
        outputs = x

    model = models.Model(inputs=inputs, outputs=outputs)

    return model


def MobileNetV1_1_0(input_shape, num_classes, include_top=True, weights=None):
    model = MobileNetV1(input_shape, num_classes, alpha=1., include_top=include_top)
    model._name = 'mobilenet_v1_1.0'

    height = input_shape[0]
    width = input_shape[1]
    if height != width or height not in [128, 160, 192, 224]:
        print("`input_shape` is undefined or non-square, "
              "or `rows` is not in [128, 160, 192, 224]. "
              "Weights for input shape (224, 224) will be"
              " loaded as the default.")
        height = 224

    if weights == 'imagenet':
        url = 'https://github.com/Runist/image-classifier-keras/releases/download/v0.2/mobilenet_1_0_{}_tf_no_top.h5'.format(height)
        weights_path = './pretrain_weights/mobilenet_1_0_{}_tf_no_top.h5'.format(height)
        if not os.path.exists(weights_path):
            print("Downloading data from {}".format(url))
            urllib.request.urlretrieve(url, weights_path, process_bar)

        model.load_weights(weights_path, by_name=True, skip_mismatch=True)

    return model


def MobileNetV1_7_5(input_shape, num_classes, include_top=True, weights=None):
    model = MobileNetV1(input_shape, num_classes, alpha=0.75, include_top=include_top)
    model._name = 'mobilenet_v1_0.75'

    height = input_shape[0]
    width = input_shape[1]
    if height != width or height not in [128, 160, 192, 224]:
        print("`input_shape` is undefined or non-square, "
              "or `rows` is not in [128, 160, 192, 224]. "
              "Weights for input shape (224, 224) will be"
              " loaded as the default.")
        height = 224

    if weights == 'imagenet':
        url = 'https://github.com/Runist/image-classifier-keras/releases/download/v0.2/mobilenet_7_5_{}_tf_no_top.h5'.format(height)
        weights_path = './pretrain_weights/mobilenet_7_5_{}_tf_no_top.h5'.format(height)
        if not os.path.exists(weights_path):
            print("Downloading data from {}".format(url))
            urllib.request.urlretrieve(url, weights_path, process_bar)

        model.load_weights(weights_path, by_name=True, skip_mismatch=True)

    return model


def MobileNetV1_5_0(input_shape, num_classes, include_top=True, weights=None):
    model = MobileNetV1(input_shape, num_classes, alpha=0.5, include_top=include_top)
    model._name = 'mobilenet_v1_0.50'

    height = input_shape[0]
    width = input_shape[1]
    if height != width or height not in [128, 160, 192, 224]:
        print("`input_shape` is undefined or non-square, "
              "or `rows` is not in [128, 160, 192, 224]. "
              "Weights for input shape (224, 224) will be"
              " loaded as the default.")
        height = 224

    if weights == 'imagenet':
        url = 'https://github.com/Runist/image-classifier-keras/releases/download/v0.2/mobilenet_5_0_{}_tf_no_top.h5'.format(height)
        weights_path = './pretrain_weights/mobilenet_5_0_{}_tf_no_top.h5'.format(height)
        if not os.path.exists(weights_path):
            print("Downloading data from {}".format(url))
            urllib.request.urlretrieve(url, weights_path, process_bar)

        model.load_weights(weights_path, by_name=True, skip_mismatch=True)

    return model


def MobileNetV1_2_5(input_shape, num_classes, include_top=True, weights=None):
    model = MobileNetV1(input_shape, num_classes, alpha=0.25, include_top=include_top)
    model._name = 'mobilenet_v1_0.25'

    height = input_shape[0]
    width = input_shape[1]
    if height != width or height not in [128, 160, 192, 224]:
        print("`input_shape` is undefined or non-square, "
              "or `rows` is not in [128, 160, 192, 224]. "
              "Weights for input shape (224, 224) will be"
              " loaded as the default.")
        height = 224

    if weights == 'imagenet':
        url = 'https://github.com/Runist/image-classifier-keras/releases/download/v0.2/mobilenet_2_5_{}_tf_no_top.h5'.format(height)
        weights_path = './pretrain_weights/mobilenet_2_5_{}_tf_no_top.h5'.format(height)
        if not os.path.exists(weights_path):
            print("Downloading data from {}".format(url))
            urllib.request.urlretrieve(url, weights_path, process_bar)

        model.load_weights(weights_path, by_name=True, skip_mismatch=True)

    return model
