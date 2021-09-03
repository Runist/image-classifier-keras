# -*- coding: utf-8 -*-
# @File : predict.py
# @Author: Runist
# @Time : 2021/7/6 14:48
# @Software: PyCharm
# @Brief: 预测脚本
from core.config import args, preprocess_dict
from core.models import get_model
import cv2 as cv
import numpy as np
import tensorflow as tf
import os


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

    preprocess = "normal"
    for key in preprocess_dict.keys():
        if key in args.network:
            preprocess = preprocess_dict[key]

    if preprocess == "normal":
        image_padded = cv.cvtColor(image_padded, cv.COLOR_BGR2RGB)
        image_padded /= 255.
    elif preprocess == "tf":
        image_padded = cv.cvtColor(image_padded, cv.COLOR_BGR2RGB)
        image_padded /= 127.5
        image_padded -= 1.
    elif preprocess == 'torch':
        image_padded = cv.cvtColor(image_padded, cv.COLOR_BGR2RGB)
        image_padded /= 255.
        image_padded -= [0.485, 0.456, 0.406]
        image_padded /= [0.229, 0.224, 0.225]
    elif preprocess == "caffe":
        image_padded -= [103.939, 116.779, 123.68]

    return image_padded


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    gpus = tf.config.experimental.list_physical_devices("GPU")
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)

    model = get_model(args.network, args.input_shape, args.num_classes)
    model.load_weights("./weights/{}/epoch=99_val_loss=0.1795_val_acc=0.9625.h5".format(args.network))
    model.trainable = False

    image = cv.imread("./dataset/daisy.jpg")
    image = image_preprocess(image, args.input_shape[:2])
    image = np.expand_dims(image, axis=0)

    result = model.predict(image)
    result = np.squeeze(result)
    index = np.argmax(result)
    score = result[index]

    print("Prediction is {}, confident score {:.4f}.".format(args.label_name[index], score))
