# -*- coding: utf-8 -*-
# @File : AlexNet.py
# @Author: Runist
# @Time : 2020/2/29 15:35
# @Software: PyCharm
# @Brief: AlexNet网络实现
import os
from tensorflow.keras import layers, models
from tensorflow.python.keras.utils import data_utils


def AlexNet(input_shape, num_classes, include_top=True, weights=None):
    """
    AlexNet网络结构
    :param input_shape: 网络输入shape
    :param num_classes: 分类数量
    :param include_top: 是否包含分类层
    :param weights: 是否有预训练权重
    :return: model
    """
    # 尝试用函数式API定义模型
    inputs = layers.Input(shape=input_shape)

    x = layers.ZeroPadding2D(((1, 2), (1, 2)))(inputs)
    x = layers.Conv2D(48, kernel_size=11, strides=4, activation='relu')(x)
    x = layers.MaxPool2D(pool_size=3, strides=2)(x)
    x = layers.Conv2D(128, kernel_size=5, padding='same', activation='relu')(x)
    x = layers.MaxPool2D(pool_size=3, strides=2)(x)
    x = layers.Conv2D(192, kernel_size=3, padding='same', activation='relu')(x)
    x = layers.Conv2D(192, kernel_size=3, padding='same', activation='relu')(x)
    x = layers.Conv2D(128, kernel_size=3, padding='same', activation='relu')(x)
    x = layers.MaxPool2D(pool_size=3, strides=2)(x)

    x = layers.Flatten()(x)
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(2048, activation='relu')(x)
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(2048, activation='relu')(x)

    if include_top:
        outputs = layers.Dense(num_classes, name='logits')(x)
    else:
        outputs = x

    model = models.Model(inputs=inputs, outputs=outputs)

    return model
