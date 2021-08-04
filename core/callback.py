# -*- coding: utf-8 -*-
# @File : callback.py
# @Author: Runist
# @Time : 2021/6/9 21:21
# @Software: PyCharm
# @Brief: lr的相关回调函数
import numpy as np
import tensorflow as tf
from tensorflow.keras import optimizers


class CosineAnnealingLRScheduler(optimizers.schedules.LearningRateSchedule):
    def __init__(self, epochs, train_step, lr_max, lr_min, warmth_rate=0.2):
        super(CosineAnnealingLRScheduler, self).__init__()
        self.total_step = epochs * train_step
        self.warm_step = int(self.total_step * warmth_rate)
        self.lr_max = lr_max
        self.lr_min = lr_min

    @tf.function
    def __call__(self, step):
        if step < self.warm_step:
            lr = self.lr_max / self.warm_step * step
        else:
            lr = self.lr_min + 0.5 * (self.lr_max - self.lr_min) * (1.0 + tf.cos((step - self.warm_step) / self.total_step * np.pi))

        return lr


def print_lr(optimizer):
    def lr(y_true, y_pred):
        return optimizer._decayed_lr("float32")
    return lr
