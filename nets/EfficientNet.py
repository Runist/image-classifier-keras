# -*- coding: utf-8 -*-
# @File : EfficientNet.py
# @Author: Runist
# @Time : 2020-11-05 11:05
# @Software: PyCharm
# @Brief: EfficientNet的网络实现
from tensorflow.keras import layers, models
from tensorflow.python.keras.applications import imagenet_utils
import collections
import math
import string
import os
import urllib.request
from core.utils import process_bar
from core.config import args

# namedtuple是一个函数，它用来创建一个自定义的tuple对象，并且规定了tuple元素的个数，并可以用属性而不是索引来引用tuple的某个元素。
BlockArgs = collections.namedtuple('BlockArgs', [
    'kernel_size', 'num_repeat', 'input_filters', 'output_filters',
    'expand_ratio', 'id_skip', 'strides', 'se_ratio'
])

# 每个MBConv的参数
DEFAULT_BLOCKS_ARGS = [
    BlockArgs(kernel_size=3, num_repeat=1, input_filters=32, output_filters=16,
              expand_ratio=1, id_skip=True, strides=1, se_ratio=0.25),
    BlockArgs(kernel_size=3, num_repeat=2, input_filters=16, output_filters=24,
              expand_ratio=6, id_skip=True, strides=2, se_ratio=0.25),
    BlockArgs(kernel_size=5, num_repeat=2, input_filters=24, output_filters=40,
              expand_ratio=6, id_skip=True, strides=2, se_ratio=0.25),
    BlockArgs(kernel_size=3, num_repeat=3, input_filters=40, output_filters=80,
              expand_ratio=6, id_skip=True, strides=2, se_ratio=0.25),
    BlockArgs(kernel_size=5, num_repeat=3, input_filters=80, output_filters=112,
              expand_ratio=6, id_skip=True, strides=1, se_ratio=0.25),
    BlockArgs(kernel_size=5, num_repeat=4, input_filters=112, output_filters=192,
              expand_ratio=6, id_skip=True, strides=2, se_ratio=0.25),
    BlockArgs(kernel_size=3, num_repeat=1, input_filters=192, output_filters=320,
              expand_ratio=6, id_skip=True, strides=1, se_ratio=0.25)
]


CONV_KERNEL_INITIALIZER = {
    'class_name': 'VarianceScaling',
    'config': {
        'scale': 2.0,
        'mode': 'fan_out',
        # EfficientNet实际上使用未截断的正态分布来初始化conv层，但是keras.initializers.VarianceScaling使用截断了的分布。
        # 我们决定使用自定义初始化程序，以实现更好的可序列化性
        'distribution': 'normal'
    }
}


DENSE_KERNEL_INITIALIZER = {
    'class_name': 'VarianceScaling',
    'config': {
        'scale': 1. / 3.,
        'mode': 'fan_out',
        'distribution': 'uniform'
    }
}


def round_filters(filters, width_coefficient, depth_divisor):
    """
    计算卷积核缩放后的数量，但要保证能被8整除
    :param filters: 原本卷积核个数
    :param width_coefficient: 网络宽度的缩放系数
    :param depth_divisor: 被除的基数
    :return: 卷积核数量
    """

    filters *= width_coefficient
    new_filters = int(filters + depth_divisor / 2) // depth_divisor * depth_divisor
    new_filters = max(depth_divisor, new_filters)

    # 确保卷积核数不低于原来的90%
    if new_filters < 0.9 * filters:
        new_filters += depth_divisor

    return int(new_filters)


def round_repeats(repeats, depth_coefficient):
    """
    基于深度乘数的重复次数的整数。
    :param repeats: 重复次数
    :param depth_coefficient: 网络深度的缩放系数
    :return:
    """
    # 向上取整
    return int(math.ceil(depth_coefficient * repeats))


def mb_conv_block(inputs, block_args, activation, drop_rate=None, prefix=''):
    """
    MB_Conv块
    :param inputs: 块输入
    :param block_args: 本次卷积块的参数
    :param activation: 激活函数
    :param drop_rate: drop的概率
    :param prefix: 名字前缀
    :return: x
    """

    has_se = (block_args.se_ratio is not None) and (0 < block_args.se_ratio <= 1)
    filters = block_args.input_filters * block_args.expand_ratio

    if block_args.expand_ratio != 1:
        x = layers.Conv2D(filters, 1,
                          padding='same',
                          use_bias=False,
                          kernel_initializer=CONV_KERNEL_INITIALIZER,
                          name=prefix + 'expand_conv')(inputs)
        x = layers.BatchNormalization(axis=-1, name=prefix + 'expand_bn')(x)
        x = layers.Activation(activation, name=prefix + 'expand_activation')(x)
    else:
        x = inputs

    if block_args.strides == 2:
        x = layers.ZeroPadding2D(
            padding=imagenet_utils.correct_pad(x, block_args.kernel_size),
            name=prefix + 'dwconv_pad')(x)
        conv_pad = 'valid'
    else:
        conv_pad = 'same'

    # Depthwise Convolution
    x = layers.DepthwiseConv2D(block_args.kernel_size,
                               strides=block_args.strides,
                               padding=conv_pad,
                               use_bias=False,
                               depthwise_initializer=CONV_KERNEL_INITIALIZER,
                               name=prefix + 'dwconv')(x)
    x = layers.BatchNormalization(axis=-1, name=prefix + 'bn')(x)
    x = layers.Activation(activation, name=prefix + 'activation')(x)

    # 压缩后再放大，作为一个调整系数
    if has_se:
        num_reduced_filters = max(1, int(block_args.input_filters * block_args.se_ratio))
        se_tensor = layers.GlobalAveragePooling2D(name=prefix + 'se_squeeze')(x)

        se_tensor = layers.Reshape((1, 1, filters), name=prefix + 'se_reshape')(se_tensor)
        se_tensor = layers.Conv2D(num_reduced_filters, 1,
                                  activation=activation,
                                  padding='same',
                                  use_bias=True,
                                  kernel_initializer=CONV_KERNEL_INITIALIZER,
                                  name=prefix + 'se_reduce')(se_tensor)
        se_tensor = layers.Conv2D(filters, 1,
                                  activation='sigmoid',
                                  padding='same',
                                  use_bias=True,
                                  kernel_initializer=CONV_KERNEL_INITIALIZER,
                                  name=prefix + 'se_expand')(se_tensor)

        x = layers.multiply([x, se_tensor], name=prefix + 'se_excite')

    # 利用1x1卷积对特征层进行压缩
    x = layers.Conv2D(block_args.output_filters, 1,
                      padding='same',
                      use_bias=False,
                      kernel_initializer=CONV_KERNEL_INITIALIZER,
                      name=prefix + 'project_conv')(x)
    x = layers.BatchNormalization(axis=-1, name=prefix + 'project_bn')(x)

    # 实现残差网络
    if block_args.id_skip and block_args.strides == 1 and block_args.input_filters == block_args.output_filters:
        if drop_rate > 0:
            x = layers.Dropout(drop_rate, noise_shape=(None, 1, 1, 1), name=prefix + 'drop')(x)

        x = layers.add([x, inputs], name=prefix + 'add')

    return x


def EfficientNet(width_coefficient,
                 depth_coefficient,
                 resolution,
                 dropout_rate=0.2,
                 drop_connect_rate=0.2,
                 depth_divisor=8,
                 blocks_args=DEFAULT_BLOCKS_ARGS,
                 input_shape=None,
                 model_name='EfficientNet',
                 num_classes=1000,
                 include_top=True):
    """
    EfficientNet模型结构
    :param width_coefficient: float，网络宽度的缩放系数
    :param depth_coefficient: float，网络深度的缩放系数
    :param resolution: int，图片分辨率
    :param dropout_rate: 最后一层前的dropout系数
    :param drop_connect_rate: 跳过连接时的概率
    :param depth_divisor: int，基数
    :param blocks_args: 用于构造块模块的BlockArgs列表
    :param input_shape: 模型输入shape
    :param model_name: string，模型名字
    :param num_classes: 分类数量
    :param include_top: 是否包含分类层
    :return: model
    """
    if input_shape:
        inputs = layers.Input(shape=input_shape)
    else:
        inputs = layers.Input(shape=(resolution, resolution, 3))
        if inputs.shape[1:] != args.input_shape:
            raise Exception("{} input shape is {}, but dataset input is {},"
                            " you must let config.py input_shape match Efficientnet input shape.".format(
                             model_name, inputs.shape[1:], args.input_shape))

    x = layers.ZeroPadding2D(
        padding=imagenet_utils.correct_pad(inputs, 3),
        name='stem_conv_pad')(inputs)

    x = layers.Conv2D(round_filters(32, width_coefficient, depth_divisor),
                      kernel_size=3,
                      strides=(2, 2),
                      padding='valid',
                      use_bias=False,
                      kernel_initializer=CONV_KERNEL_INITIALIZER,
                      name='stem_conv')(x)
    x = layers.BatchNormalization(axis=-1, name='stem_bn')(x)
    x = layers.Activation('swish', name='stem_activation')(x)

    # 计算MBConv总共重复的次数
    num_blocks_total = sum(block_args.num_repeat for block_args in blocks_args)
    block_num = 0

    for idx, block_args in enumerate(blocks_args):

        # 根据深度乘法器更新块输入和输出卷积核个数
        block_args = block_args._replace(
            input_filters=round_filters(block_args.input_filters, width_coefficient, depth_divisor),
            output_filters=round_filters(block_args.output_filters, width_coefficient, depth_divisor),
            num_repeat=round_repeats(block_args.num_repeat, depth_coefficient))

        # 逐层增加drop_rate的概率
        drop_rate = drop_connect_rate * float(block_num) / num_blocks_total
        # 第一个MBConv块需要注意步长和过滤器尺寸的增加
        x = mb_conv_block(x, block_args,
                          activation='swish',
                          drop_rate=drop_rate,
                          prefix='block{}a_'.format(idx + 1))

        block_num += 1
        if block_args.num_repeat > 1:
            # 因为前面修改过卷积核的个数，所以后面的卷积核个数也需要修改，保证MBConv卷积最后输入输出一样
            block_args = block_args._replace(input_filters=block_args.output_filters, strides=1)

            for b_idx in range(block_args.num_repeat - 1):
                drop_rate = drop_connect_rate * float(block_num) / num_blocks_total
                block_prefix = 'block{}{}_'.format(idx + 1, string.ascii_lowercase[b_idx + 1])

                x = mb_conv_block(x,
                                  block_args,
                                  activation='swish',
                                  drop_rate=drop_rate,
                                  prefix=block_prefix)
                block_num += 1

    x = layers.Conv2D(round_filters(1280, width_coefficient, depth_divisor),
                      kernel_size=1,
                      padding='same',
                      use_bias=False,
                      kernel_initializer=CONV_KERNEL_INITIALIZER,
                      name='top_conv')(x)

    x = layers.BatchNormalization(axis=-1, name='top_bn')(x)
    x = layers.Activation('swish', name='top_activation')(x)

    # 利用GlobalAveragePooling2D代替全连接层
    x = layers.GlobalAveragePooling2D(name='avg_pool')(x)
    if dropout_rate and dropout_rate > 0:
        x = layers.Dropout(dropout_rate, name='top_dropout')(x)

    if include_top:
        outputs = layers.Dense(num_classes,
                               kernel_initializer=DENSE_KERNEL_INITIALIZER,
                               activation="softmax",
                               name="prediction")(x)
    else:
        outputs = x

    model = models.Model(inputs, outputs, name=model_name)

    return model


def EfficientNetB0(num_classes,
                   alpha=1.0, beta=1.0, r=224, input_shape=None,
                   include_top=False, weights=None):

    model = EfficientNet(alpha, beta, r, 0.2,
                         model_name='efficientnet-b0',
                         input_shape=input_shape,
                         num_classes=num_classes,
                         include_top=include_top)

    if weights == 'imagenet':
        url = 'https://github.com/Runist/image-classifier-keras/releases/download/v0.2/efficientnetb0_notop.h5'
        weights_path = './pretrain_weights/efficientnetb0_notop.h5'
        if not os.path.exists(weights_path):
            print("Downloading data from {}".format(url))
            urllib.request.urlretrieve(url, weights_path, process_bar)

        model.load_weights(weights_path, by_name=True, skip_mismatch=True)

    return model


def EfficientNetB1(num_classes,
                   alpha=1.0, beta=1.1, r=240, input_shape=None,
                   include_top=True, weights=None):

    model = EfficientNet(alpha, beta, r, 0.2,
                         model_name='efficientnet-b1',
                         input_shape=input_shape,
                         num_classes=num_classes,
                         include_top=include_top)

    if weights == 'imagenet':
        url = 'https://github.com/Runist/image-classifier-keras/releases/download/v0.2/efficientnetb1_notop.h5'
        weights_path = './pretrain_weights/efficientnetb1_notop.h5'
        if not os.path.exists(weights_path):
            print("Downloading data from {}".format(url))
            urllib.request.urlretrieve(url, weights_path, process_bar)

        model.load_weights(weights_path, by_name=True, skip_mismatch=True)

    return model


def EfficientNetB2(num_classes,
                   alpha=1.1, beta=1.2, r=260, input_shape=None,
                   include_top=True, weights=None):

    model = EfficientNet(alpha, beta, r, 0.3,
                         model_name='efficientnet-b2',
                         input_shape=input_shape,
                         num_classes=num_classes,
                         include_top=include_top)

    if weights == 'imagenet':
        url = 'https://github.com/Runist/image-classifier-keras/releases/download/v0.2/efficientnetb2_notop.h5'
        weights_path = './pretrain_weights/efficientnetb2_notop.h5'
        if not os.path.exists(weights_path):
            print("Downloading data from {}".format(url))
            urllib.request.urlretrieve(url, weights_path, process_bar)

        model.load_weights(weights_path, by_name=True, skip_mismatch=True)

    return model


def EfficientNetB3(num_classes,
                   alpha=1.2, beta=1.4, r=300, input_shape=None,
                   include_top=True, weights=None):

    model = EfficientNet(alpha, beta, r, 0.3,
                         model_name='efficientnet-b3',
                         input_shape=input_shape,
                         num_classes=num_classes,
                         include_top=include_top)

    if weights == 'imagenet':
        url = 'https://github.com/Runist/image-classifier-keras/releases/download/v0.2/efficientnetb3_notop.h5'
        weights_path = './pretrain_weights/efficientnetb3_notop.h5'
        if not os.path.exists(weights_path):
            print("Downloading data from {}".format(url))
            urllib.request.urlretrieve(url, weights_path, process_bar)

        model.load_weights(weights_path, by_name=True, skip_mismatch=True)

    return model


def EfficientNetB4(num_classes,
                   alpha=1.4, beta=1.8, r=380, input_shape=None,
                   include_top=True, weights=None):

    model = EfficientNet(alpha, beta, r, 0.4,
                         model_name='efficientnet-b4',
                         input_shape=input_shape,
                         num_classes=num_classes,
                         include_top=include_top)

    if weights == 'imagenet':
        url = 'https://github.com/Runist/image-classifier-keras/releases/download/v0.2/efficientnetb4_notop.h5'
        weights_path = './pretrain_weights/efficientnetb4_notop.h5'
        if not os.path.exists(weights_path):
            print("Downloading data from {}".format(url))
            urllib.request.urlretrieve(url, weights_path, process_bar)

        model.load_weights(weights_path, by_name=True, skip_mismatch=True)

    return model


def EfficientNetB5(num_classes,
                   alpha=1.6, beta=2.2, r=456, input_shape=None,
                   include_top=True, weights=None):

    model = EfficientNet(alpha, beta, r, 0.4,
                         model_name='efficientnet-b5',
                         input_shape=input_shape,
                         num_classes=num_classes,
                         include_top=include_top)

    if weights == 'imagenet':
        url = 'https://github.com/Runist/image-classifier-keras/releases/download/v0.2/efficientnetb5_notop.h5'
        weights_path = './pretrain_weights/efficientnetb5_notop.h5'
        if not os.path.exists(weights_path):
            print("Downloading data from {}".format(url))
            urllib.request.urlretrieve(url, weights_path, process_bar)

        model.load_weights(weights_path, by_name=True, skip_mismatch=True)

    return model


def EfficientNetB6(num_classes,
                   alpha=1.8, beta=2.6, r=528, input_shape=None,
                   include_top=True, weights=None):

    model = EfficientNet(alpha, beta, r, 0.5,
                         model_name='efficientnet-b6',
                         input_shape=input_shape,
                         num_classes=num_classes,
                         include_top=include_top)

    if weights == 'imagenet':
        url = 'https://github.com/Runist/image-classifier-keras/releases/download/v0.2/efficientnetb6_notop.h5'
        weights_path = './pretrain_weights/efficientnetb6_notop.h5'
        if not os.path.exists(weights_path):
            print("Downloading data from {}".format(url))
            urllib.request.urlretrieve(url, weights_path, process_bar)

        model.load_weights(weights_path, by_name=True, skip_mismatch=True)

    return model


def EfficientNetB7(num_classes,
                   alpha=2.0, beta=3.1, r=600, input_shape=None,
                   include_top=True, weights=None):

    model = EfficientNet(alpha, beta, r, 0.5,
                         model_name='efficientnet-b7',
                         input_shape=input_shape,
                         num_classes=num_classes,
                         include_top=include_top)

    if weights == 'imagenet':
        url = 'https://github.com/Runist/image-classifier-keras/releases/download/v0.2/efficientnetb7_notop.h5'
        weights_path = './pretrain_weights/efficientnetb7_notop.h5'
        if not os.path.exists(weights_path):
            print("Downloading data from {}".format(url))
            urllib.request.urlretrieve(url, weights_path, process_bar)

        model.load_weights(weights_path, by_name=True, skip_mismatch=True)

    return model


def customEfficientNet(num_classes, alpha, beta, r, input_shape=None, include_top=True):
    return EfficientNet(
        alpha, beta, r, 0.2,
        model_name='customefficientnet',
        input_shape=input_shape,
        num_classes=num_classes,
        include_top=include_top)
