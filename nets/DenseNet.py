# -*- coding: utf-8 -*-
# @File : DenseNet.py
# @Author: Runist
# @Time : 2020/8/24 9:17
# @Software: PyCharm
# @Brief: DenseNet网络实现

import os
import urllib.request
from tensorflow.keras import layers, models
from core.utils import process_bar


def conv_block(x, growth_rate, name, dropout_rate=None):
    """
    DenseNet的conv块，以论文的描述是BN-ReLU-Conv
    :param x: 输入变量
    :param growth_rate: 增长率
    :param name: conv块的名字前缀
    :param dropout_rate: dropout的比率
    :return: x
    """
    x1 = layers.BatchNormalization(epsilon=1.001e-5, name=name+'_0_bn')(x)
    x1 = layers.Activation('relu', name=name+'_0_relu')(x1)
    x1 = layers.Conv2D(4 * growth_rate, kernel_size=1, use_bias=False, name=name+'_1_conv')(x1)
    if dropout_rate:
        x1 = layers.Dropout(dropout_rate, name=name+'_0_dropout')(x1)

    x1 = layers.BatchNormalization(epsilon=1.001e-5, name=name+'_1_bn')(x1)
    x1 = layers.Activation('relu', name=name+'_1_relu')(x1)
    x1 = layers.Conv2D(growth_rate, padding='same', kernel_size=3, use_bias=False, name=name+'_2_conv')(x1)
    if dropout_rate:
        x1 = layers.Dropout(dropout_rate, name=name+'_1_dropout')(x1)

    x = layers.Concatenate(name=name+'_concat')([x, x1])

    return x


def transition_block(x, reduction, name):
    """
    过渡层，每个Dense Block直接降采样的部分
    :param x: 输入
    :param reduction: 维度降低的部分
    :param name: transition_block的名字去前缀
    :return: x
    """
    x = layers.BatchNormalization(epsilon=1.001e-5, name=name+'_bn')(x)
    x = layers.Activation('relu', name=name+'_relu')(x)
    # 降维
    x = layers.Conv2D(int(x.shape[-1] * reduction), kernel_size=1, use_bias=False, name=name+'_conv')(x)
    x = layers.AveragePooling2D(2, strides=2, name=name+'_pool')(x)

    return x


def dense_block(x, blocks, growth_rate, name, dropout_rate=None):
    """
    一个dense block由多个卷积块组成
    :param x: 输入
    :param blocks: 每个dense block卷积多少次
    :param growth_rate: 每个特征层的增长率
    :param name: conv块的名字前缀
    :param dropout_rate: dropout的比率
    :return: x
    """
    for i in range(blocks):
        x = conv_block(x, growth_rate, name=name+'_block'+str(i+1), dropout_rate=dropout_rate)
    return x


def DenseNet(input_shape, blocks, num_classes, include_top=True, growth_rate=32, reduction=0.5, dropout_rate=None):
    """
    建立DenseNet网络，需要调节dense block的数量、一个dense block中有多少个conv、growth_rate、reduction、dropout rate
    :param input_shape: 网络输入shape
    :param blocks: 卷积块的数量
    :param num_classes: 分类数量
    :param include_top: 是否包含分类层
    :param growth_rate: 每个特征层的增长率
    :param reduction: 过渡层减少层数的比例
    :param dropout_rate: dropout的比率
    :return: model
    """
    inputs = layers.Input(shape=input_shape)

    x = layers.ZeroPadding2D(padding=((3, 3), (3, 3)))(inputs)
    x = layers.Conv2D(64, 7, strides=2, use_bias=False, name='conv1/conv')(x)
    x = layers.BatchNormalization(epsilon=1.001e-5, name='conv1/bn')(x)
    x = layers.Activation('relu', name='conv1/relu')(x)

    x = layers.ZeroPadding2D(padding=((1, 1), (1, 1)))(x)
    x = layers.MaxPooling2D(3, strides=2, name='pool1')(x)

    x = dense_block(x, blocks[0], growth_rate=growth_rate, name='conv2', dropout_rate=dropout_rate)
    x = transition_block(x, reduction=reduction, name='pool2')
    x = dense_block(x, blocks[1], growth_rate=growth_rate, name='conv3', dropout_rate=dropout_rate)
    x = transition_block(x, reduction=reduction, name='pool3')
    x = dense_block(x, blocks[2], growth_rate=growth_rate, name='conv4', dropout_rate=dropout_rate)
    x = transition_block(x, reduction=reduction, name='pool4')
    x = dense_block(x, blocks[3], growth_rate=growth_rate, name='conv5', dropout_rate=dropout_rate)

    x = layers.BatchNormalization(epsilon=1.001e-5, name='bn')(x)
    x = layers.Activation('relu', name='relu')(x)

    x = layers.GlobalAveragePooling2D(name='avg_pool')(x)

    if include_top:
        outputs = layers.Dense(num_classes, name="prediction", activation="softmax")(x)
    else:
        outputs = x

    model = models.Model(inputs, outputs)

    return model


def DenseNet121(input_shape, num_classes, include_top=True, growth_rate=32, reduction=0.5, dropout_rate=0., weights=None):
    model = DenseNet(input_shape, [6, 12, 24, 16], num_classes,
                     include_top=include_top,
                     growth_rate=growth_rate,
                     reduction=reduction,
                     dropout_rate=dropout_rate)
    model._name = 'densenet121'

    if weights == 'imagenet':
        url = 'https://github.com/Runist/image-classifier-keras/releases/download/v0.2/densenet121_weights_tf_dim_ordering_tf_kernels_notop.h5'
        weights_path = './pretrain_weights/densenet121_weights_tf_dim_ordering_tf_kernels_notop.h5'
        if not os.path.exists(weights_path):
            print("Downloading data from {}".format(url))
            urllib.request.urlretrieve(url, weights_path, process_bar)

        model.load_weights(weights_path, by_name=True, skip_mismatch=True)

    return model


def DenseNet169(input_shape, num_classes, include_top=True, growth_rate=32, reduction=0.5, dropout_rate=0., weights=None):
    model = DenseNet(input_shape, [6, 12, 32, 32], num_classes,
                     include_top=include_top,
                     growth_rate=growth_rate,
                     reduction=reduction,
                     dropout_rate=dropout_rate)
    model._name = 'densenet169'

    if weights == 'imagenet':
        url = 'https://github.com/Runist/image-classifier-keras/releases/download/v0.2/densenet169_weights_tf_dim_ordering_tf_kernels_notop.h5'
        weights_path = './pretrain_weights/densenet169_weights_tf_dim_ordering_tf_kernels_notop.h5'
        if not os.path.exists(weights_path):
            print("Downloading data from {}".format(url))
            urllib.request.urlretrieve(url, weights_path, process_bar)
        model.load_weights(weights_path, by_name=True, skip_mismatch=True)

    return model


def DenseNet201(input_shape, num_classes, include_top=True, growth_rate=32, reduction=0.5, dropout_rate=0., weights=None):
    model = DenseNet(input_shape, [6, 12, 48, 32], num_classes,
                     include_top=include_top,
                     growth_rate=growth_rate,
                     reduction=reduction,
                     dropout_rate=dropout_rate)
    model._name = 'densenet201'

    if weights == 'imagenet':
        url = 'https://github.com/Runist/image-classifier-keras/releases/download/v0.2/densenet201_weights_tf_dim_ordering_tf_kernels_notop.h5'
        weights_path = './pretrain_weights/densenet201_weights_tf_dim_ordering_tf_kernels_notop.h5'
        if not os.path.exists(weights_path):
            print("Downloading data from {}".format(url))
            urllib.request.urlretrieve(url, weights_path, process_bar)

        model.load_weights(weights_path, by_name=True, skip_mismatch=True)

    return model


def DenseNet264(input_shape, num_classes, include_top=True, growth_rate=32, reduction=0.5, dropout_rate=0., weights=None):
    model = DenseNet(input_shape, [6, 12, 64, 48], num_classes,
                     include_top=include_top,
                     growth_rate=growth_rate,
                     reduction=reduction,
                     dropout_rate=dropout_rate)
    model._name = 'densenet264'

    return model

