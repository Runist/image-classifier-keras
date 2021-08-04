# -*- coding: utf-8 -*-
# @File : models.py
# @Author: Runist
# @Time : 2021/7/6 23:07
# @Software: PyCharm
# @Brief: 模型调用

import os
from nets.AlexNet import AlexNet
from nets.GoogLeNet import GoogLeNet
from nets.VGG import VGG13, VGG16, VGG19
from nets.ResNet import ResNet50, ResNet101, ResNet152
from nets.DenseNet import DenseNet121, DenseNet169, DenseNet201, DenseNet264
from nets.MobileNet_v1 import MobileNetV1_1_0, MobileNetV1_7_5, MobileNetV1_5_0, MobileNetV1_2_5
from nets.MobileNet_v2 import MobileNetV2_1_4, MobileNetV2_1_3, MobileNetV2_1_0, MobileNetV2_7_5, MobileNetV2_5_0, MobileNetV2_3_5
from nets.SEResNet import SE_ResNet50, SE_ResNet101, SE_ResNet152
from tensorflow.keras import layers, models


def get_model(network, input_shape, num_classes, include_top=True):
    if not os.path.exists("pretrain_weights"):
        os.mkdir("pretrain_weights")

    if network == "alexnet":
        model = AlexNet(input_shape, num_classes, include_top=include_top, weights=None)
    elif network == "googlenet" or network == "inception_v1":
        model = GoogLeNet(input_shape, num_classes, include_top=include_top, aux=False, weights=None)
    elif network == "vgg13":
        model = VGG13(input_shape, num_classes, include_top=include_top, weights=None)
    elif network == "vgg16":
        model = VGG16(input_shape, num_classes, include_top=include_top, weights="imagenet")
    elif network == "vgg19":
        model = VGG19(input_shape, num_classes, include_top=include_top, weights="imagenet")
    elif network == "resnet50":
        model = ResNet50(input_shape, num_classes, include_top=include_top, weights="imagenet")
    elif network == "resnet101":
        model = ResNet101(input_shape, num_classes, include_top=include_top, weights="imagenet")
    elif network == "resnet152":
        model = ResNet152(input_shape, num_classes, include_top=include_top, weights="imagenet")
    elif network == "densenet121":
        model = DenseNet121(input_shape, num_classes, include_top=include_top, weights="imagenet")
    elif network == "densenet169":
        model = DenseNet169(input_shape, num_classes, include_top=include_top, weights="imagenet")
    elif network == "densenet201":
        model = DenseNet201(input_shape, num_classes, include_top=include_top, weights="imagenet")
    elif network == "densenet264":
        model = DenseNet264(input_shape, num_classes, include_top=include_top, weights=None)
    elif network == "mobilenet_v1_1.0":
        model = MobileNetV1_1_0(input_shape, num_classes, include_top=include_top, weights="imagenet")
    elif network == "mobilenet_v1_0.75":
        model = MobileNetV1_7_5(input_shape, num_classes, include_top=include_top, weights="imagenet")
    elif network == "mobilenet_v1_0.50":
        model = MobileNetV1_5_0(input_shape, num_classes, include_top=include_top, weights="imagenet")
    elif network == "mobilenet_v1_0.25":
        model = MobileNetV1_2_5(input_shape, num_classes, include_top=include_top, weights="imagenet")
    elif network == "mobilenet_v2_1.4":
        model = MobileNetV2_1_4(input_shape, num_classes, include_top=include_top, weights="imagenet")
    elif network == "mobilenet_v2_1.3":
        model = MobileNetV2_1_3(input_shape, num_classes, include_top=include_top, weights="imagenet")
    elif network == "mobilenet_v2_1.0":
        model = MobileNetV2_1_0(input_shape, num_classes, include_top=include_top, weights="imagenet")
    elif network == "mobilenet_v2_0.75":
        model = MobileNetV2_7_5(input_shape, num_classes, include_top=include_top, weights="imagenet")
    elif network == "mobilenet_v2_0.50":
        model = MobileNetV2_5_0(input_shape, num_classes, include_top=include_top, weights="imagenet")
    elif network == "mobilenet_v2_0.35":
        model = MobileNetV2_3_5(input_shape, num_classes, include_top=include_top, weights="imagenet")
    elif network == "se_resnet50":
        model = SE_ResNet50(input_shape, num_classes, include_top=include_top, weights="imagenet")
    elif network == "se_resnet101":
        model = SE_ResNet101(input_shape, num_classes, include_top=include_top, weights="imagenet")
    elif network == "se_resnet152":
        model = SE_ResNet152(input_shape, num_classes, include_top=include_top, weights="imagenet")
    else:
        raise Exception("You don't select any model. Check config.py network name.")

    if not include_top:
        x = model.layers[-1].output
        x = layers.Dropout(rate=0.5)(x)
        x = layers.Dense(1024)(x)
        x = layers.Dropout(rate=0.5)(x)
        outputs = layers.Dense(num_classes, name="logits")(x)
    else:
        outputs = model.layers[-1].output

    model = models.Model(inputs=model.inputs, outputs=outputs, name=model.name)

    return model
