# -*- coding: utf-8 -*-
# @File : MobileNet_v2.py
# @Author: Runist
# @Time : 2020/6/10 11:31
# @Software: PyCharm
# :Brief: MobileNet v2的实现

import os
import urllib.request
from tensorflow.keras import layers, models, applications
from core.utils import process_bar


def inverted_res_block(inputs, filters, alpha, stride, expand_ratio, block_id):
    """
    倒残差结构
    :param inputs: 输入特征层
    :param filters: 卷积数量
    :param alpha: 卷积数量
    :param stride: 步长
    :param expand_ratio: 倍乘因子
    :param block_id: 此残差块的序号
    :return: 输出特征层
    """

    # 倍乘率是决定中间的倒残差结构的通道数
    in_channel = inputs.shape[-1]
    out_channel = int(filters * alpha)
    out_channel = make_divisible(out_channel, 8)

    prefix = "block_{}_".format(block_id)
    x = inputs

    # V2在DW卷积之前新加了一个PW卷积。这么做的原因，是因为 DW 卷积由于本身的计算特性决定它自己没有改变通道数的能力，
    # 上一层给它多少通道，它就只能输出多少通道。所以如果上一层给的通道数本身很少的话，DW也只能很委屈的在低维空间提特征，因此效果不够好。
    # 现在V2为了改善这个问题，给每个DW之前都配备了一个PW，专门用来升维，定义升维系数为6，
    # 这样不管输入通道数是多是少，经过第一个PW升维之后，DW都是在相对的更高维进行着辛勤工作的。
    if block_id:
        x = layers.Conv2D(expand_ratio * in_channel, 1, padding='same', use_bias=False, name=prefix + "expand")(x)
        x = layers.BatchNormalization(momentum=0.999, epsilon=1e-3, name=prefix + 'expand_BN')(x)
        x = layers.ReLU(6.0, name=prefix + 'expand_relu')(x)
    else:
        prefix = 'expanded_conv_'

    if stride == 2:
        x = layers.ZeroPadding2D(padding=((0, 1), (0, 1)), name=prefix + 'pad')(x)

    # 3x3 depthwise conv
    x = layers.DepthwiseConv2D(kernel_size=3,
                               padding="same" if stride == 1 else 'valid',
                               strides=stride,
                               use_bias=False,
                               name=prefix + 'depthwise')(x)
    x = layers.BatchNormalization(momentum=0.999,
                                  epsilon=1e-3,
                                  name=prefix + "depthwise_BN")(x)
    x = layers.ReLU(6.0, name=prefix + "depthwise_relu")(x)

    # 1x1 pointwise conv 不采用激活函数
    # 这么做的原因，是因为作者认为激活函数在高维空间能够有效的增加非线性，而在低维空间时则会破坏特征，不如线性的效果好。
    # 由于第二个PW的主要功能就是降维，因此按照上面的理论，降维之后就不宜再使用激活函数了。
    x = layers.Conv2D(filters=out_channel,
                      kernel_size=1,
                      padding="same",
                      use_bias=False,
                      name=prefix + "project")(x)
    x = layers.BatchNormalization(momentum=0.999, epsilon=1e-3, name=prefix + "project_BN")(x)

    # 满足两个条件才能使用short cut
    if stride == 1 and in_channel == out_channel:
        return layers.Add(name=prefix + "add")([inputs, x])

    return x


def make_divisible(v, divisor, min_value=None):
    """
    保证在用alpha调节网络宽度时,卷积数目时divisor的倍数
    :param v: 原来的卷积数目
    :param divisor: 基数
    :param min_value: 计算后的最小值
    :return: new_v
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


def MobileNetV2(input_shape, alpha):
    """
    MobileNetV2版本，相比MobileNetV1是在每个DW卷积、PW卷积之前多了一个通道递增的卷积，
    然后再将卷积后的结果与输入融合所以看起来是中间高，两边低
    :param input_shape: 网络输入shape
    :param alpha: 减少网络宽度的系数，0 < alpha < 1
    :return: model
    """
    inputs = layers.Input(shape=input_shape, dtype="float32")

    first_filters = make_divisible(32 * alpha, 8)

    x = layers.ZeroPadding2D(padding=((1, 0), (1, 0)), name='Conv1_pad')(inputs)
    x = layers.Conv2D(filters=first_filters, kernel_size=3, strides=(2, 2), use_bias=False, name="Conv1")(x)
    x = layers.BatchNormalization(momentum=0.999, epsilon=1e-3, name='bn_Conv1')(x)
    x = layers.ReLU(6.0, name='Conv1_relu')(x)

    x = inverted_res_block(x, 16, alpha, 1, 1, block_id=0)
    x = inverted_res_block(x, 24, alpha, 2, 6, block_id=1)
    x = inverted_res_block(x, 24, alpha, 1, 6, block_id=2)
    x = inverted_res_block(x, 32, alpha, 2, 6, block_id=3)
    x = inverted_res_block(x, 32, alpha, 1, 6, block_id=4)
    x = inverted_res_block(x, 32, alpha, 1, 6, block_id=5)
    x = inverted_res_block(x, 64, alpha, 2, 6, block_id=6)
    x = inverted_res_block(x, 64, alpha, 1, 6, block_id=7)
    x = inverted_res_block(x, 64, alpha, 1, 6, block_id=8)
    x = inverted_res_block(x, 64, alpha, 1, 6, block_id=9)
    x = inverted_res_block(x, 96, alpha, 1, 6, block_id=10)
    x = inverted_res_block(x, 96, alpha, 1, 6, block_id=11)
    x = inverted_res_block(x, 96, alpha, 1, 6, block_id=12)
    x = inverted_res_block(x, 160, alpha, 2, 6, block_id=13)
    x = inverted_res_block(x, 160, alpha, 1, 6, block_id=14)
    x = inverted_res_block(x, 160, alpha, 1, 6, block_id=15)
    x = inverted_res_block(x, 320, alpha, 1, 6, block_id=16)

    if alpha > 1.0:
        last_filters = make_divisible(1280 * alpha, 8)
    else:
        last_filters = 1280

    x = layers.Conv2D(filters=last_filters, kernel_size=1, use_bias=False, name="Conv_1")(x)
    x = layers.BatchNormalization(momentum=0.999, epsilon=1e-3, name='Conv_1_bn')(x)
    x = layers.ReLU(6.0, name='out_relu')(x)

    x = layers.GlobalAveragePooling2D()(x)

    model = models.Model(inputs=inputs, outputs=x)

    return model


def MobileNetV2_1_4(input_shape, num_classes, include_top=True, weights=None):
    model = MobileNetV2(input_shape, alpha=1.4)

    height = input_shape[0]
    width = input_shape[1]
    if height != width or height not in [96, 128, 160, 192, 224]:
        print("`input_shape` is undefined or non-square, "
              "or `rows` is not in [96, 128, 160, 192, 224]. "
              "Weights for input shape (224, 224) will be"
              " loaded as the default.")
        height = 224

    if weights == 'imagenet':
        url = 'https://github.com/Runist/image-classifier-keras/releases/download/v0.2/mobilenet_v2_weights_tf_dim_ordering_tf_kernels_1.4_{}_no_top.h5'.format(height)
        weights_path = './pretrain_weights/mobilenet_v2_weights_tf_dim_ordering_tf_kernels_1.4_{}_no_top.h5'.format(height)
        if not os.path.exists(weights_path):
            print("Downloading data from {}".format(url))
            urllib.request.urlretrieve(url, weights_path, process_bar)

        model.load_weights(weights_path)

    x = model.layers[-1].output
    if include_top:
        outputs = layers.Dense(num_classes, name="prediction", activation="softmax")(x)
    else:
        outputs = x
    model = models.Model(inputs=model.inputs, outputs=outputs, name='mobilenet_v2_1.4')

    return model


def MobileNetV2_1_3(input_shape, num_classes, include_top=True, weights=None):
    model = MobileNetV2(input_shape, alpha=1.3)

    height = input_shape[0]
    width = input_shape[1]
    if height != width or height not in [96, 128, 160, 192, 224]:
        print("`input_shape` is undefined or non-square, "
              "or `rows` is not in [96, 128, 160, 192, 224]. "
              "Weights for input shape (224, 224) will be"
              " loaded as the default.")
        height = 224

    if weights == 'imagenet':
        url = 'https://github.com/Runist/image-classifier-keras/releases/download/v0.2/mobilenet_v2_weights_tf_dim_ordering_tf_kernels_1.3_{}_no_top.h5'.format(height)
        weights_path = './pretrain_weights/mobilenet_v2_weights_tf_dim_ordering_tf_kernels_1.3_{}_no_top.h5'.format(height)
        if not os.path.exists(weights_path):
            print("Downloading data from {}".format(url))
            urllib.request.urlretrieve(url, weights_path, process_bar)

        model.load_weights(weights_path)

    x = model.layers[-1].output
    if include_top:
        outputs = layers.Dense(num_classes, name="prediction", activation="softmax")(x)
    else:
        outputs = x
    model = models.Model(inputs=model.inputs, outputs=outputs, name='mobilenet_v2_1.3')

    return model


def MobileNetV2_1_0(input_shape, num_classes, include_top=True, weights=None):
    model = MobileNetV2(input_shape, alpha=1.)

    height = input_shape[0]
    width = input_shape[1]
    if height != width or height not in [96, 128, 160, 192, 224]:
        print("`input_shape` is undefined or non-square, "
              "or `rows` is not in [128, 160, 192, 224]. "
              "Weights for input shape (224, 224) will be"
              " loaded as the default.")
        height = 224

    if weights == 'imagenet':
        url = 'https://github.com/Runist/image-classifier-keras/releases/download/v0.2/mobilenet_v2_weights_tf_dim_ordering_tf_kernels_1.0_{}_no_top.h5'.format(height)
        weights_path = './pretrain_weights/mobilenet_v2_weights_tf_dim_ordering_tf_kernels_1.0_{}_no_top.h5'.format(height)
        if not os.path.exists(weights_path):
            print("Downloading data from {}".format(url))
            urllib.request.urlretrieve(url, weights_path, process_bar)

        model.load_weights(weights_path)

    x = model.layers[-1].output
    if include_top:
        outputs = layers.Dense(num_classes, name="prediction", activation="softmax")(x)
    else:
        outputs = x
    model = models.Model(inputs=model.inputs, outputs=outputs, name='mobilenet_v2_1.0')

    return model


def MobileNetV2_7_5(input_shape, num_classes, include_top=True, weights=None):
    model = MobileNetV2(input_shape, alpha=0.75)

    height = input_shape[0]
    width = input_shape[1]
    if height != width or height not in [96, 128, 160, 192, 224]:
        print("`input_shape` is undefined or non-square, "
              "or `rows` is not in [128, 160, 192, 224]. "
              "Weights for input shape (224, 224) will be"
              " loaded as the default.")
        height = 224

    if weights == 'imagenet':
        url = 'https://github.com/Runist/image-classifier-keras/releases/download/v0.2/mobilenet_v2_weights_tf_dim_ordering_tf_kernels_0.75_{}_no_top.h5'.format(height)
        weights_path = './pretrain_weights/mobilenet_v2_weights_tf_dim_ordering_tf_kernels_0.75_{}_no_top.h5'.format(height)
        if not os.path.exists(weights_path):
            print("Downloading data from {}".format(url))
            urllib.request.urlretrieve(url, weights_path, process_bar)

        model.load_weights(weights_path)

    x = model.layers[-1].output
    if include_top:
        outputs = layers.Dense(num_classes, name="prediction", activation="softmax")(x)
    else:
        outputs = x
    model = models.Model(inputs=model.inputs, outputs=outputs, name='mobilenet_v2_0.75')

    return model


def MobileNetV2_5_0(input_shape, num_classes, include_top=True, weights=None):
    model = MobileNetV2(input_shape, alpha=0.5)

    height = input_shape[0]
    width = input_shape[1]
    if height != width or height not in [128, 160, 192, 224]:
        print("`input_shape` is undefined or non-square, "
              "or `rows` is not in [96, 128, 160, 192, 224]. "
              "Weights for input shape (224, 224) will be"
              " loaded as the default.")
        height = 224

    if weights == 'imagenet':
        url = 'https://github.com/Runist/image-classifier-keras/releases/download/v0.2/mobilenet_v2_weights_tf_dim_ordering_tf_kernels_0.5_{}_no_top.h5'.format(height)
        weights_path = './pretrain_weights/mobilenet_v2_weights_tf_dim_ordering_tf_kernels_0.5_{}_no_top.h5'.format(height)
        if not os.path.exists(weights_path):
            print("Downloading data from {}".format(url))
            urllib.request.urlretrieve(url, weights_path, process_bar)

        model.load_weights(weights_path)

    x = model.layers[-1].output
    if include_top:
        outputs = layers.Dense(num_classes, name="prediction", activation="softmax")(x)
    else:
        outputs = x
    model = models.Model(inputs=model.inputs, outputs=outputs, name='mobilenet_v2_0.50')

    return model


def MobileNetV2_3_5(input_shape, num_classes, include_top=True, weights=None):
    model = MobileNetV2(input_shape, alpha=0.35)

    height = input_shape[0]
    width = input_shape[1]
    if height != width or height not in [96, 128, 160, 192, 224]:
        print("`input_shape` is undefined or non-square, "
              "or `rows` is not in [96, 128, 160, 192, 224]. "
              "Weights for input shape (224, 224) will be"
              " loaded as the default.")
        height = 224

    if weights == 'imagenet':
        url = 'https://github.com/Runist/image-classifier-keras/releases/download/v0.2/mobilenet_v2_weights_tf_dim_ordering_tf_kernels_0.35_{}_no_top.h5'.format(height)
        weights_path = './pretrain_weights/mobilenet_v2_weights_tf_dim_ordering_tf_kernels_0.35_{}_no_top.h5'.format(height)
        if not os.path.exists(weights_path):
            print("Downloading data from {}".format(url))
            urllib.request.urlretrieve(url, weights_path, process_bar)

        model.load_weights(weights_path)

    x = model.layers[-1].output
    if include_top:
        outputs = layers.Dense(num_classes, name="prediction", activation="softmax")(x)
    else:
        outputs = x
    model = models.Model(inputs=model.inputs, outputs=outputs, name='mobilenet_v2_0.35')

    return model
