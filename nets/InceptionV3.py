# -*- coding: utf-8 -*-
# @File : InceptionV3.py
# @Author: Runist
# @Time : 2021/8/16 17:42
# @Software: PyCharm
# @Brief: InceptionV3网络实现

import os
import urllib.request
from tensorflow.keras import layers, models, applications
from core.utils import process_bar


def conv2d_bn_relu(inputs, filters, kernel_row, kernel_col, padding='same', strides=(1, 1), name=None):
    """
    conv + BN + relu
    :param inputs: 输入tensor
    :param filters: 卷积核数量
    :param kernel_row: 行卷积的数量
    :param kernel_col: 列卷积的数量
    :param padding: padding的方式
    :param strides: 卷积步长
    :param name: 名字前缀
    :return: x
    """

    if name is not None:
        bn_name = name + '_bn'
        conv_name = name + '_conv'
    else:
        bn_name = None
        conv_name = None

    x = layers.Conv2D(filters, (kernel_row, kernel_col), strides=strides, padding=padding,
                      use_bias=False, name=conv_name)(inputs)
    # scale是控制是否使用归一化之后的线性变化gamma值，center是控制beta值
    x = layers.BatchNormalization(scale=False, name=bn_name)(x, training=False)
    x = layers.Activation('relu', name=name)(x)

    return x


def InceptionV3(input_shape, num_classes, include_top=True, weights=None):
    """
    InceptionV3网络构建
    :param input_shape: 网络输入shape
    :param num_classes: 分类数量
    :param include_top: 是否包含分类层
    :param weights: 是否使用预训练权重
    :return: model
    """
    inputs = layers.Input(shape=input_shape)

    x = conv2d_bn_relu(inputs, 32, 3, 3, strides=(2, 2), padding='valid')
    x = conv2d_bn_relu(x, 32, 3, 3, padding='valid')
    x = conv2d_bn_relu(x, 64, 3, 3)
    x = layers.MaxPooling2D((3, 3), strides=(2, 2))(x)

    x = conv2d_bn_relu(x, 80, 1, 1, padding='valid')
    x = conv2d_bn_relu(x, 192, 3, 3, padding='valid')
    x = layers.MaxPooling2D((3, 3), strides=(2, 2))(x)

    # mixed 0: 35 x 35 x 256
    branch1x1 = conv2d_bn_relu(x, 64, 1, 1)

    branch5x5 = conv2d_bn_relu(x, 48, 1, 1)
    branch5x5 = conv2d_bn_relu(branch5x5, 64, 5, 5)

    branch3x3dbl = conv2d_bn_relu(x, 64, 1, 1)
    branch3x3dbl = conv2d_bn_relu(branch3x3dbl, 96, 3, 3)
    branch3x3dbl = conv2d_bn_relu(branch3x3dbl, 96, 3, 3)

    branch_pool = layers.AveragePooling2D((3, 3), strides=(1, 1), padding='same')(x)
    branch_pool = conv2d_bn_relu(branch_pool, 32, 1, 1)
    x = layers.concatenate([branch1x1, branch5x5, branch3x3dbl, branch_pool], name='mixed0')

    # mixed 1, 2: 35 x 35 x 288
    for i in range(2):
        branch1x1 = conv2d_bn_relu(x, 64, 1, 1)

        branch5x5 = conv2d_bn_relu(x, 48, 1, 1)
        branch5x5 = conv2d_bn_relu(branch5x5, 64, 5, 5)

        branch3x3dbl = conv2d_bn_relu(x, 64, 1, 1)
        branch3x3dbl = conv2d_bn_relu(branch3x3dbl, 96, 3, 3)
        branch3x3dbl = conv2d_bn_relu(branch3x3dbl, 96, 3, 3)

        branch_pool = layers.AveragePooling2D((3, 3), strides=(1, 1), padding='same')(x)
        branch_pool = conv2d_bn_relu(branch_pool, 64, 1, 1)
        x = layers.concatenate([branch1x1, branch5x5, branch3x3dbl, branch_pool], name='mixed' + str(i + 1))

    # mixed 3: 17 x 17 x 768
    branch3x3 = conv2d_bn_relu(x, 384, 3, 3, strides=(2, 2), padding='valid')

    branch3x3dbl = conv2d_bn_relu(x, 64, 1, 1)
    branch3x3dbl = conv2d_bn_relu(branch3x3dbl, 96, 3, 3)
    branch3x3dbl = conv2d_bn_relu(branch3x3dbl, 96, 3, 3, strides=(2, 2), padding='valid')

    branch_pool = layers.MaxPooling2D((3, 3), strides=(2, 2))(x)
    x = layers.concatenate([branch3x3, branch3x3dbl, branch_pool], name='mixed3')

    # mixed 4: 17 x 17 x 768
    branch1x1 = conv2d_bn_relu(x, 192, 1, 1)

    branch7x7 = conv2d_bn_relu(x, 128, 1, 1)
    branch7x7 = conv2d_bn_relu(branch7x7, 128, 1, 7)
    branch7x7 = conv2d_bn_relu(branch7x7, 192, 7, 1)

    branch7x7dbl = conv2d_bn_relu(x, 128, 1, 1)
    branch7x7dbl = conv2d_bn_relu(branch7x7dbl, 128, 7, 1)
    branch7x7dbl = conv2d_bn_relu(branch7x7dbl, 128, 1, 7)
    branch7x7dbl = conv2d_bn_relu(branch7x7dbl, 128, 7, 1)
    branch7x7dbl = conv2d_bn_relu(branch7x7dbl, 192, 1, 7)

    branch_pool = layers.AveragePooling2D((3, 3), strides=(1, 1), padding='same')(x)
    branch_pool = conv2d_bn_relu(branch_pool, 192, 1, 1)
    x = layers.concatenate([branch1x1, branch7x7, branch7x7dbl, branch_pool], name='mixed4')

    # mixed 5, 6: 17 x 17 x 768
    for i in range(2):
        branch1x1 = conv2d_bn_relu(x, 192, 1, 1)

        branch7x7 = conv2d_bn_relu(x, 160, 1, 1)
        branch7x7 = conv2d_bn_relu(branch7x7, 160, 1, 7)
        branch7x7 = conv2d_bn_relu(branch7x7, 192, 7, 1)

        branch7x7dbl = conv2d_bn_relu(x, 160, 1, 1)
        branch7x7dbl = conv2d_bn_relu(branch7x7dbl, 160, 7, 1)
        branch7x7dbl = conv2d_bn_relu(branch7x7dbl, 160, 1, 7)
        branch7x7dbl = conv2d_bn_relu(branch7x7dbl, 160, 7, 1)
        branch7x7dbl = conv2d_bn_relu(branch7x7dbl, 192, 1, 7)

        branch_pool = layers.AveragePooling2D((3, 3), strides=(1, 1), padding='same')(x)
        branch_pool = conv2d_bn_relu(branch_pool, 192, 1, 1)
        x = layers.concatenate([branch1x1, branch7x7, branch7x7dbl, branch_pool], name='mixed' + str(5 + i))

    # mixed 7: 17 x 17 x 768
    branch1x1 = conv2d_bn_relu(x, 192, 1, 1)

    branch7x7 = conv2d_bn_relu(x, 192, 1, 1)
    branch7x7 = conv2d_bn_relu(branch7x7, 192, 1, 7)
    branch7x7 = conv2d_bn_relu(branch7x7, 192, 7, 1)

    branch7x7dbl = conv2d_bn_relu(x, 192, 1, 1)
    branch7x7dbl = conv2d_bn_relu(branch7x7dbl, 192, 7, 1)
    branch7x7dbl = conv2d_bn_relu(branch7x7dbl, 192, 1, 7)
    branch7x7dbl = conv2d_bn_relu(branch7x7dbl, 192, 7, 1)
    branch7x7dbl = conv2d_bn_relu(branch7x7dbl, 192, 1, 7)

    branch_pool = layers.AveragePooling2D((3, 3), strides=(1, 1), padding='same')(x)
    branch_pool = conv2d_bn_relu(branch_pool, 192, 1, 1)
    x = layers.concatenate([branch1x1, branch7x7, branch7x7dbl, branch_pool], name='mixed7')

    # mixed 8: 8 x 8 x 1280
    branch3x3 = conv2d_bn_relu(x, 192, 1, 1)
    branch3x3 = conv2d_bn_relu(branch3x3, 320, 3, 3, strides=(2, 2), padding='valid')

    branch7x7x3 = conv2d_bn_relu(x, 192, 1, 1)
    branch7x7x3 = conv2d_bn_relu(branch7x7x3, 192, 1, 7)
    branch7x7x3 = conv2d_bn_relu(branch7x7x3, 192, 7, 1)
    branch7x7x3 = conv2d_bn_relu(branch7x7x3, 192, 3, 3, strides=(2, 2), padding='valid')

    branch_pool = layers.MaxPooling2D((3, 3), strides=(2, 2))(x)
    x = layers.concatenate([branch3x3, branch7x7x3, branch_pool], name='mixed8')

    # mixed 9: 8 x 8 x 2048
    for i in range(2):
        branch1x1 = conv2d_bn_relu(x, 320, 1, 1)

        branch3x3 = conv2d_bn_relu(x, 384, 1, 1)
        branch3x3_1 = conv2d_bn_relu(branch3x3, 384, 1, 3)
        branch3x3_2 = conv2d_bn_relu(branch3x3, 384, 3, 1)
        branch3x3 = layers.concatenate([branch3x3_1, branch3x3_2], name='mixed9_' + str(i))

        branch3x3dbl = conv2d_bn_relu(x, 448, 1, 1)
        branch3x3dbl = conv2d_bn_relu(branch3x3dbl, 384, 3, 3)
        branch3x3dbl_1 = conv2d_bn_relu(branch3x3dbl, 384, 1, 3)
        branch3x3dbl_2 = conv2d_bn_relu(branch3x3dbl, 384, 3, 1)
        branch3x3dbl = layers.concatenate([branch3x3dbl_1, branch3x3dbl_2])

        branch_pool = layers.AveragePooling2D((3, 3), strides=(1, 1), padding='same')(x)
        branch_pool = conv2d_bn_relu(branch_pool, 192, 1, 1)
        x = layers.concatenate([branch1x1, branch3x3, branch3x3dbl, branch_pool], name='mixed' + str(9 + i))

    x = layers.GlobalAvgPool2D(name='avg_pool')(x)

    if include_top:
        outputs = layers.Dense(num_classes, name="logits", activation="softmax")(x)
    else:
        outputs = x

    model = models.Model(inputs=inputs, outputs=outputs, name="inception_v3")

    if weights == 'imagenet':
        url = 'https://github.com/Runist/image-classifier-keras/releases/download/v0.2/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5'
        weights_path = '../pretrain_weights/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5'

        if not os.path.exists(weights_path):
            print("Downloading data from {}".format(url))
            urllib.request.urlretrieve(url, weights_path, process_bar)

        model.load_weights(weights_path, by_name=True, skip_mismatch=True)

    return model


