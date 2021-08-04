# -*- coding: utf-8 -*-
# @File : config.py
# @Author: Runist
# @Time : 2021/7/2 14:17
# @Software: PyCharm
# @Brief: 配置文件

train_data_path = "./dataset/train"
test_data_path = "./dataset/validation"

label_name = [
    "daisy",
    "dandelion",
    "roses",
    "sunflowers",
    "tulips"
]

input_shape = (224, 224, 3)
num_classes = len(label_name)
batch_size = 32

network = "resnet50"
lr_init = 1e-4
lr_end = 1e-5
first_epochs = 50
second_epochs = 50
