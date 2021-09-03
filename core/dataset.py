# -*- coding: utf-8 -*-
# @File : dataset.py
# @Author: Runist
# @Time : 2021/7/2 14:17
# @Software: PyCharm
# @Brief: 数据读取脚本
import os
import glob
import random
import cv2 as cv
import numpy as np
import tensorflow as tf
from PIL import Image, ImageEnhance
from core.config import args, preprocess_dict


class Dataset:
    def __init__(self, data_path, label_name, batch_size=4, target_size=(224, 224), aug=False, pretrain=True):

        self.batch_size = batch_size
        self.target_size = target_size
        self.data_path = data_path
        self.aug = aug
        self.label_name = label_name
        self.num_classes = len(self.label_name)
        # 根据不同网络选择对应的预处理方法
        for key in preprocess_dict.keys():
            if key in args.network:
                self.preprocess = preprocess_dict[key]
        # 如果不使用imagenet的预训练权重，则置为使用常规归一化
        if not pretrain:
            self.preprocess = "normal"

        assert os.path.exists(self.data_path), "Can't find {}".format(self.data_path)
        self.set_image_info()

    def __len__(self):
        return len(self.image_info)

    def set_image_info(self):
        """
        继承自Dataset类，需要实现对输入图像路径的读取和mask路径的读取，且存储到self.image_info中
        :return:
        """
        if not os.path.exists(self.data_path):
            Exception("{} is not exists!".format(self.data_path))

        self.image_info = []

        for label_index, label_name in enumerate(self.label_name):
            label_path = os.path.join(self.data_path, label_name)
            image_group = glob.glob("{}/*.jpg".format(label_path))

            for image_path in image_group:
                self.image_info.append({"image_path": image_path, "label": label_index})

    def read_image(self, image_id):
        """
        读取图像
        :param image_id: self.image_info的数字索引
        :return: image
        """
        image_path = self.image_info[image_id]["image_path"]
        image = cv.imread(image_path)

        return image

    def read_label(self, image_id):
        """
        读取label
        :param image_id: self.image_info的数字索引
        :return: image
        """
        label_index = self.image_info[image_id]["label"]
        label = tf.one_hot(label_index, depth=self.num_classes)

        return label

    def preprocess_input(self, image):
        if self.preprocess == "normal":
            image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
            image /= 255.
        elif self.preprocess == "tf":
            image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
            image /= 127.5
            image -= 1.
        elif self.preprocess == 'torch':
            image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
            image /= 255.
            image -= [0.485, 0.456, 0.406]
            image /= [0.229, 0.224, 0.225]
        elif self.preprocess == "caffe":
            image -= [103.939, 116.779, 123.68]

        return image

    def resize_image(self, image, pad_value=None):
        """
        resize图像，多余的地方用其他颜色填充
        :param image: 输入图像
        :param pad_value: 填充区域像素值
        :return: image_padded
        """
        if pad_value is None:
            image_resize = cv.resize(image, self.target_size, interpolation=cv.INTER_LINEAR)
            return image_resize

        image_h, image_w = image.shape[:2]
        input_h, input_w = self.target_size

        scale = min(input_h / image_h, input_w / image_w)

        image_h = int(image_h * scale)
        image_w = int(image_w * scale)

        dw, dh = (input_w - image_w) // 2, (input_h - image_h) // 2

        # image 用双线性插值
        image_resize = cv.resize(image, (image_w, image_h), interpolation=cv.INTER_LINEAR)
        image_padded = np.full(shape=[input_h, input_w, 3], fill_value=pad_value)
        image_padded[dh: image_h+dh, dw: image_w+dw, :] = image_resize

        return image_padded

    def random_horizontal_flip(self, image):
        """
        左右翻转图像
        :param image: 输入图像
        :return:
        """
        _, w, _ = image.shape
        image = cv.flip(image, 1)

        return image

    def random_crop(self, image):
        """
        随机裁剪
        :param image: 输入图像
        :return:
        """
        h, w, _ = image.shape

        max_l_trans = w // 10
        max_u_trans = h // 10
        max_r_trans = w - w // 10
        max_d_trans = h - h // 10

        crop_xmin = int(random.uniform(0, max_l_trans))
        crop_ymin = int(random.uniform(0, max_u_trans))
        crop_xmax = int(random.uniform(max_r_trans, w))
        crop_ymax = int(random.uniform(max_d_trans, h))

        image = image[crop_ymin: crop_ymax, crop_xmin: crop_xmax]

        return image

    def random_translate(self, image):
        """
        整图随机位移
        :param image: 输入图像
        :return:
        """

        h, w, _ = image.shape

        max_l_trans = h // 10
        max_u_trans = w // 10
        max_r_trans = h // 10
        max_d_trans = w // 10

        tx = random.uniform(-(max_l_trans - 1), (max_r_trans - 1))
        ty = random.uniform(-(max_u_trans - 1), (max_d_trans - 1))

        M = np.array([[1, 0, tx], [0, 1, ty]])

        image = cv.warpAffine(image, M, (w, h), borderValue=(128, 128, 128))

        return image

    def color_jitter(self, image, hue=0.1, sat=1.5, val=1.5):
        """
        色域抖动数据增强
        :param image: 输入图像
        :param hue: 色调
        :param sat: 饱和度
        :param val: 明度
        :return: image
        """
        image = np.array(image, np.float32) / 255
        image = cv.cvtColor(image, cv.COLOR_BGR2HSV)

        h = random.uniform(-hue, hue)
        s = random.uniform(1, sat) if random.random() < .5 else 1/random.uniform(1, sat)
        v = random.uniform(1, val) if random.random() < .5 else 1/random.uniform(1, val)

        image[..., 0] += h * 360
        image[..., 0][image[..., 0] > 1] -= 1.
        image[..., 0][image[..., 0] < 0] += 1.
        image[..., 1] *= s
        image[..., 2] *= v
        image[image[:, :, 0] > 360, 0] = 360
        image[:, :, 1:][image[:, :, 1:] > 1] = 1
        image[image < 0] = 0

        image = cv.cvtColor(image, cv.COLOR_HSV2BGR) * 255
        image = image.astype(np.uint8)

        return image

    def random_brightness(self, image, brightness_range):
        """
        随机亮度加减
        :param image: 输入图像
        :param brightness_range: 亮度加减范围
        :return: image
        """
        image = np.array(image, np.float32) / 255
        hsv = cv.cvtColor(image, cv.COLOR_BGR2HSV)
        h, s, v = cv.split(hsv)

        value = random.uniform(-brightness_range, brightness_range)

        v += value
        v[v > 1] = 1.
        v[v < 0] = 0.

        final_hsv = cv.merge((h, s, v))
        image = cv.cvtColor(final_hsv, cv.COLOR_HSV2BGR) * 255
        image = image.astype(np.uint8)

        return image

    def random_sharpness(self, image, sharp_range=3.):
        """
        随机锐度加强
        :param image: 输入图像
        :param sharp_range: 锐度加减范围
        :return: image
        """
        image = Image.fromarray(image)
        enh_sha = ImageEnhance.Sharpness(image)
        image = enh_sha.enhance(random.uniform(-0.5, sharp_range))
        image = np.array(image)

        return image

    def parse(self, index):
        """
        tf.data的解析器
        :param index: 字典索引
        :return:
        """

        def get_data(i):
            image = self.read_image(i)
            label = self.read_label(i)

            if random.random() < 0.4 and self.aug:
                if random.random() < 0.5 and self.aug:
                    image = self.random_horizontal_flip(image)
                if random.random() < 0.5 and self.aug:
                    image = self.color_jitter(image)
                if random.random() < 0.5 and self.aug:
                    image = self.random_brightness(image, brightness_range=0.3)
                if random.random() < 0.5 and self.aug:
                    image = self.random_sharpness(image, sharp_range=3.)
                if random.random() < 0.5 and self.aug:
                    image = self.random_crop(image)
                if random.random() < 0.5 and self.aug:
                    image = self.random_translate(image)

            image = self.resize_image(image, pad_value=128.)
            image = image.astype(np.float32)

            image = self.preprocess_input(image)

            return image, label

        image, label = tf.py_function(get_data, [index], [tf.float32, tf.float32])
        h, w = self.target_size

        image.set_shape([h, w, 3])
        label.set_shape([self.num_classes, ])

        return image, label

    def tf_dataset(self):
        """
        用tf.data的方式读取数据，以提高gpu使用率
        :return: 数据集
        """
        index = [i for i in range(len(self))]
        # 这是GPU读取方式
        dataset = tf.data.Dataset.from_tensor_slices(index)

        dataset = dataset.map(self.parse, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        dataset = dataset.repeat().batch(self.batch_size)
        dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)

        return dataset

