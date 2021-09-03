# -*- coding: utf-8 -*-
# @File : GoogLeNet.py
# @Author: Runist
# @Time : 2020/3/2 15:41
# @Software: PyCharm
# @Brief: GoogLeNet的网络实现
import os
import tensorflow as tf
from tensorflow.keras import layers, models, backend
from tensorflow.python.keras.utils import data_utils


class LRN(layers.Layer):

    def __init__(self, alpha=0.0001, k=1, beta=0.75, n=5, **kwargs):
        self.alpha = alpha
        self.k = k
        self.beta = beta
        self.n = n
        super(LRN, self).__init__(**kwargs)

    def call(self, x, mask=None):
        b, ch, r, c = x.shape
        half_n = self.n // 2                  # half the local region
        input_sqr = backend.square(x)         # square the input

        input_sqr = tf.pad(input_sqr, [[0, 0], [half_n, half_n], [0, 0], [0, 0]])

        scale = self.k                      # offset for the scale
        norm_alpha = self.alpha / self.n    # normalized alpha
        for i in range(self.n):
            scale += norm_alpha * input_sqr[:, i:i+ch, :, :]
        scale = scale ** self.beta
        x = x / scale

        return x

    def get_config(self):
        config = {"alpha": self.alpha,
                  "k": self.k,
                  "beta": self.beta,
                  "n": self.n}

        base_config = super(LRN, self).get_config()

        return dict(list(base_config.items()) + list(config.items()))


def inception(inputs, filter1, filter2_1, filter2_3, filter3_1, filter3_5, filter4, name):
    """
    inception结构
    :param inputs: 输入tensor
    :param filter1: 最左边1x1的卷积
    :param filter2_1: 第二个分支1x1的卷积核数量
    :param filter2_3: 第二个分支3x3的卷积核数量
    :param filter3_1: 第三个分支1x1的卷积核数量
    :param filter3_5: 第三个分支5x5的卷积核数量
    :param filter4: 第四个分支1x1的卷积核数量
    :param name: inception卷积块的名字前缀
    :return: outputs
    """
    x1 = layers.Conv2D(filter1, kernel_size=1, activation='relu', name=name+'/1x1')(inputs)

    x2 = layers.Conv2D(filter2_1, kernel_size=1, activation='relu', name=name+'/3x3_reduce')(inputs)
    x2 = layers.Conv2D(filter2_3, kernel_size=3, padding='SAME', activation='relu', name=name+'/3x3')(x2)

    x3 = layers.Conv2D(filter3_1, kernel_size=1, activation='relu', name=name+'/5x5_reduce')(inputs)
    x3 = layers.Conv2D(filter3_5, kernel_size=5, padding='SAME', activation='relu', name=name+'/5x5')(x3)

    x4 = layers.MaxPool2D(pool_size=3, strides=1, padding='SAME', name=name+'/pool')(inputs)
    x4 = layers.Conv2D(filter4, kernel_size=1, activation='relu', name=name+'/pool_proj')(x4)

    outputs = layers.Concatenate(axis=-1, name=name+'/output')([x1, x2, x3, x4])

    return outputs


def inception_aux(inputs, num_classes, name):
    """
    inception结构
    :param inputs: 输入tensor
    :param num_classes: 分类数量
    :param name: inception卷积块的名字前缀
    :return: outputs
    """
    x = layers.AvgPool2D(pool_size=5, strides=3, name=name+'/ave_pool')(inputs)
    x = layers.Conv2D(128, kernel_size=1, activation="relu", name=name+'/conv')(x)

    x = layers.Dense(1024, activation='relu', name=name+'/fc')(x)
    outputs = layers.Dense(num_classes, name=name+'/classifier')(x)

    # outputs = layers.Softmax()(x)

    return outputs


def GoogLeNet(input_shape, num_classes, include_top=True, aux=False, weights=None):
    """
    GoogLeNet又称Inception v1
    :param input_shape: 网络输入shape
    :param num_classes: 分类的数量
    :param include_top: 是否包含分类层
    :param aux: 是否使用辅助分类器
    :param weights: 是否有预训练权重
    :return:
    """
    inputs = layers.Input(shape=input_shape)
    x = layers.Conv2D(64, kernel_size=7, strides=2, padding='SAME', activation='relu')(inputs)
    x = layers.MaxPool2D(pool_size=3, strides=2, padding='SAME', name='maxpool_1')(x)
    x = LRN(name='pool1/norm1')(x)

    x = layers.Conv2D(64, kernel_size=1, strides=1, activation='relu')(x)
    x = layers.Conv2D(192, kernel_size=3, strides=1, padding='SAME', activation='relu')(x)
    x = LRN(name='conv2/norm2')(x)
    x = layers.MaxPool2D(pool_size=3, strides=2, padding='SAME', name='maxpool_2')(x)

    x = inception(x, 64, 96, 128, 16, 32, 32, name='inception_3a')
    x = inception(x, 128, 128, 192, 32, 96, 64, name='inception_3b')
    x = layers.MaxPool2D(pool_size=3, strides=2, padding='SAME', name='maxpool_3')(x)

    x = inception(x, 192, 96, 208, 16, 48, 64, name='inception_4a')
    # 辅助分类器1
    if aux:
        aux1 = inception_aux(x, num_classes, name='loss1')

    x = inception(x, 160, 112, 224, 24, 64, 64, name='inception_4b')
    x = inception(x, 128, 128, 256, 24, 64, 64, name='inception_4c')
    x = inception(x, 112, 144, 288, 32, 64, 64, name='inception_4d')
    if aux:
        aux2 = inception_aux(x, num_classes, name='loss2')

    x = inception(x, 256, 160, 320, 32, 128, 128, name='inception_4e')
    x = layers.MaxPool2D(pool_size=3, strides=2, padding='SAME', name='maxpool_4')(x)

    x = inception(x, 256, 160, 320, 32, 128, 128, name='inception_5a')
    x = inception(x, 384, 192, 384, 48, 128, 128, name='inception_5b')

    x = layers.AvgPool2D(pool_size=7, strides=1, name='avgpool')(x)

    x = layers.Flatten(name='output_flatten')(x)
    x = layers.Dropout(rate=0.4, name='output_dropout')(x)

    if include_top:
        outputs = layers.Dense(num_classes, name='loss3/classifier', activation="softmax")(x)
    else:
        outputs = x

    if aux:
        # 辅助分类器的初衷是用来防止梯度弥散和梯度爆炸的，但是在实际训练中用处不大
        model = models.Model(inputs=inputs, outputs=[aux1, aux2, outputs], name="googlenet")
    else:
        model = models.Model(inputs=inputs, outputs=outputs, name="googlenet")

    # TODO 转换权重
    # if weights:
    #     model.load_weights("../pretrain_weights/googlenet_weights.h5", by_name=True, skip_mismatch=True)

    return model

