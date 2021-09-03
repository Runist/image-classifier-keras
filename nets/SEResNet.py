# -*- coding: utf-8 -*-
# @File : SEResNet.py
# @Author: Runist
# @Time : 2020-12-30 10:05
# @Software: PyCharm
# @Brief: SENet网络实现

import os
import urllib.request
from tensorflow.keras import layers, models
from core.utils import process_bar


def se_identity_block(inputs, filters, name, strides=1, reduce_ratio=8, training=True):
    """
    se_resnet卷积块
    :param inputs: 卷积块的输入
    :param filters: 卷积核的数量
    :param name: 卷积块名字相关
    :param strides: 为1时候不改变特征层宽高，为2就减半
    :param reduce_ratio: se block的衰减倍数，用于减少参数量
    :param training: 是否是训练模式
    :return : x
    """
    filter1, filter2, filter3 = filters

    x = layers.Conv2D(filter1, kernel_size=1, strides=strides, name=name+'_1_conv')(inputs)
    x = layers.BatchNormalization(name=name+'_1_bn')(x, training=training)
    x = layers.ReLU(name=name + '_1_relu')(x)

    x = layers.Conv2D(filter2, kernel_size=3, strides=strides, padding='same', name=name+'_2_conv')(x)
    x = layers.BatchNormalization(name=name+'_2_bn')(x, training=training)
    x = layers.ReLU(name=name + '_2_relu')(x)

    x = layers.Conv2D(filter3, kernel_size=1, strides=strides, name=name+'_3_conv')(x)
    x = layers.BatchNormalization(name=name+'_3_bn')(x, training=training)

    se_input = x
    x = layers.GlobalAveragePooling2D(name=name+'avg')(se_input)
    x = layers.Reshape((1, 1, filter3), name=name+'reshape')(x)
    x = layers.Dense(filter3 // reduce_ratio, activation='relu', name=name+'dense_relu')(x)
    x = layers.Dense(filter3, activation='sigmoid', name=name+'dense_sigmoid')(x)
    x = layers.Multiply(name=name+'multiply')([x, se_input])

    x = layers.Add(name=name + '_add')([x, inputs])
    x = layers.ReLU(name=name + '_out')(x)

    return x


def se_conv_block(inputs, filters, name, strides=1, reduce_ratio=8, training=True):
    """
    se_bottleneck卷积块
    :param inputs: 卷积块的输入
    :param filters: 卷积核的数量
    :param name: 卷积块名字相关
    :param strides: 为1时候不改变特征层宽高，为2就减半
    :param reduce_ratio: se block的衰减倍数，用于减少参数量
    :param training: 是否是训练模式
    :return : x
    """
    filter1, filter2, filter3 = filters

    shortcut = layers.Conv2D(filter3, kernel_size=1, strides=strides, name=name+'_0_conv')(inputs)
    shortcut = layers.BatchNormalization(name=name+'_0_bn')(shortcut, training=training)

    x = layers.Conv2D(filter1, kernel_size=1, strides=strides, name=name+'_1_conv')(inputs)
    x = layers.BatchNormalization(name=name+'_1_bn')(x, training=training)
    x = layers.ReLU(name=name + '_1_relu')(x)

    x = layers.Conv2D(filter2, kernel_size=3, padding='same', name=name+'_2_conv')(x)
    x = layers.BatchNormalization(name=name+'_2_bn')(x, training=training)
    x = layers.ReLU(name=name + '_2_relu')(x)

    x = layers.Conv2D(filter3, kernel_size=1, name=name+'_3_conv')(x)
    x = layers.BatchNormalization(name=name+'_3_bn')(x, training=training)

    se_input = x
    x = layers.GlobalAveragePooling2D(name=name+'avg')(se_input)
    x = layers.Reshape((1, 1, filter3), name=name+'reshape')(x)
    x = layers.Dense(filter3 // reduce_ratio, activation='relu', name=name+'dense_relu')(x)
    x = layers.Dense(filter3, activation='sigmoid', name=name+'dense_sigmoid')(x)
    x = layers.Multiply(name=name+'multiply')([x, se_input])

    x = layers.Add(name=name + '_add')([x, shortcut])
    x = layers.ReLU(name=name + '_out')(x)

    return x


def SE_ResNet_stage(inputs, filters, num_block, name, strides):
    """
    ResNet中一个stage结构
    :param inputs: stage的输入
    :param filters: 每个卷积块对应卷积核的数量
    :param num_block: 卷积块重复的数量
    :param name: 该卷积块的名字前缀
    :param strides: 步长
    :return: x
    """
    x = se_conv_block(inputs, filters, name=name+'_block1', strides=strides)
    for i in range(1, num_block):
        x = se_identity_block(x, filters, name=name+'_block'+str(i+1))

    return x


def SE_ResNet(input_shape, num_classes, blocks_list, include_top=True):
    """
    ResNet网络结构，通过传入不同的残差块和重复的次数进行不同层数的ResNet构建
    :param input_shape: 网络输入shape
    :param num_classes: 分类数量
    :param blocks_list: 每个残差单元重复的次数列表
    :param include_top: 是否包含分类层
    :return: model
    """
    inputs = layers.Input(shape=input_shape)

    x = layers.ZeroPadding2D(padding=((3, 3), (3, 3)), name='conv1_pad')(inputs)

    x = layers.Conv2D(filters=64, kernel_size=7, strides=2, name='conv1_conv')(x)
    x = layers.BatchNormalization(momentum=0.9, epsilon=1e-5, name='conv1_bn')(x)
    x = layers.ReLU(name='conv1_relu')(x)

    x = layers.ZeroPadding2D(padding=((1, 1), (1, 1)), name='pool1_pad')(x)
    x = layers.MaxPool2D(pool_size=3, strides=2, name="pool1_pool")(x)

    x = SE_ResNet_stage(x, [64, 64, 256], blocks_list[0], name='conv2', strides=1)
    x = SE_ResNet_stage(x, [128, 128, 512], blocks_list[1], name='conv3', strides=2)
    x = SE_ResNet_stage(x, [256, 256, 1024], blocks_list[2], name='conv4', strides=2)
    x = SE_ResNet_stage(x, [512, 512, 2048], blocks_list[3], name='conv5', strides=2)

    x = layers.GlobalAvgPool2D(name='avg_pool')(x)

    if include_top:
        outputs = layers.Dense(num_classes, name="prediction", activation="softmax")(x)
    else:
        outputs = x

    model = models.Model(inputs=inputs, outputs=outputs)

    return model


# def SE_ResNet18(height, width, num_classes):
#     return SE_ResNet(height, width, num_classes, SEBasicBlock, [2, 2, 2, 2])
#
#
# def SE_ResNet34(height, width, num_classes):
#     return SE_ResNet(height, width, num_classes, SEBasicBlock, [3, 4, 6, 3])


def SE_ResNet50(input_shape, num_classes, include_top=True, weights=None):
    model = SE_ResNet(input_shape, num_classes, [3, 4, 6, 3], include_top=include_top)
    model._name = 'se_resnet50'

    if weights == 'imagenet':
        url = 'https://github.com/Runist/image-classifier-keras/releases/download/v0.2/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5'
        weights_path = './pretrain_weights/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5'
        if not os.path.exists(weights_path):
            print("Downloading data from {}".format(url))
            urllib.request.urlretrieve(url, weights_path, process_bar)

        model.load_weights(weights_path, by_name=True, skip_mismatch=True)

    return model


def SE_ResNet101(input_shape, num_classes, include_top=True, weights=None):
    model = SE_ResNet(input_shape, num_classes, [3, 4, 23, 3], include_top=include_top)
    model._name = 'se_resnet101'

    if weights == 'imagenet':
        url = 'https://github.com/Runist/image-classifier-keras/releases/download/v0.2/resnet101_weights_tf_dim_ordering_tf_kernels_notop.h5'
        weights_path = './pretrain_weights/resnet101_weights_tf_dim_ordering_tf_kernels_notop.h5'
        if not os.path.exists(weights_path):
            print("Downloading data from {}".format(url))
            urllib.request.urlretrieve(url, weights_path, process_bar)

        model.load_weights(weights_path, by_name=True, skip_mismatch=True)

    return model


def SE_ResNet152(input_shape, num_classes, include_top=True, weights=None):
    model = SE_ResNet(input_shape, num_classes, [3, 8, 36, 3], include_top=include_top)
    model._name = 'se_resnet152'

    if weights == 'imagenet':
        url = 'https://github.com/Runist/image-classifier-keras/releases/download/v0.2/resnet152_weights_tf_dim_ordering_tf_kernels_notop.h5'
        weights_path = './pretrain_weights/resnet152_weights_tf_dim_ordering_tf_kernels_notop.h5'
        if not os.path.exists(weights_path):
            print("Downloading data from {}".format(url))
            urllib.request.urlretrieve(url, weights_path, process_bar)

        model.load_weights(weights_path, by_name=True, skip_mismatch=True)

    return model
