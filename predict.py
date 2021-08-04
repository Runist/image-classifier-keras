# -*- coding: utf-8 -*-
# @File : predict.py
# @Author: Runist
# @Time : 2021/7/6 14:48
# @Software: PyCharm
# @Brief:
import core.config as cfg
from core.models import get_model
import cv2 as cv
import numpy as np
import tensorflow as tf


def image_preprocess(image, target_size, pad_value=128.0):
    """
    resize图像，多余的地方用其他颜色填充
    :param image: 输入图像
    :param pad_value: 填充区域像素值
    :return: image_padded
    """
    image_h, image_w = image.shape[:2]
    input_h, input_w = target_size

    scale = min(input_h / image_h, input_w / image_w)

    image_h = int(image_h * scale)
    image_w = int(image_w * scale)

    dw, dh = (input_w - image_w) // 2, (input_h - image_h) // 2

    # image 用双线性插值
    image_resize = cv.resize(image, (image_w, image_h), interpolation=cv.INTER_LINEAR)
    image_padded = np.full(shape=[input_h, input_w, 3], fill_value=pad_value)
    image_padded[dh: image_h+dh, dw: image_w+dw, :] = image_resize
    image_padded = image_padded.astype(np.float32)
    image_padded -= [123.68, 116.68, 103.94]

    return image_padded


if __name__ == '__main__':
    model = get_model(cfg.network, cfg.input_shape, cfg.num_classes)
    model.load_weights("./weights/{}/epoch=40_val_loss=0.3655_val_acc=0.9148.h5".format(cfg.network))

    image = cv.imread("./dataset/daisy.jpg")
    image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
    image = image_preprocess(image, cfg.input_shape[:2])
    image = np.expand_dims(image, axis=0)

    result = model.predict(image)
    result = tf.nn.softmax(result)
    result = np.squeeze(result)
    index = np.argmax(result)
    score = result[index]

    print("Prediction is {}, confident score {:.4f}.".format(cfg.label_name[index], score))
