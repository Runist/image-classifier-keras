# -*- coding: utf-8 -*-
# @File : config.py
# @Author: Runist
# @Time : 2021/7/2 14:17
# @Software: PyCharm
# @Brief: 配置文件

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=str, default='1', help='Select gpu device.')
parser.add_argument('--train_image_dir', type=str, default="./dataset/train",
                    help='The directory containing the train image data.')
parser.add_argument('--test_image_dir', type=str, default="./dataset/validation",
                    help='The directory containing the validation image data.')
parser.add_argument('--label_name', type=list, default=[
    "daisy",
    "dandelion",
    "roses",
    "sunflowers",
    "tulips"
], help='The name of class.')
parser.add_argument('--input_shape', type=tuple, default=(224, 224, 3),
                    help='The image shape of model input.')
parser.add_argument('--num_classes', type=int, default=5,
                    help='The number of class.')
parser.add_argument('--include_top', type=bool, default=True,
                    help='Whether the classification layer is included.')
parser.add_argument('--batch_size', type=int, default=32,
                    help='Number of examples per batch.')
parser.add_argument('--network', type=str, default="resnet50",
                    choices=["alexnet", "googlenet", "vgg13", "vgg16", "vgg19", "resnet50", "resnet101", "resnet152",
                             "densenet121", "densenet169", "densenet201", "densenet264", "inception_v3", "xception",
                             "mobilenet_v1_1.0", "mobilenet_v1_0.75", "mobilenet_v1_0.50", "mobilenet_v1_0.25",
                             "mobilenet_v2_1.4", "mobilenet_v2_1.3", "mobilenet_v2_1.0", "mobilenet_v2_0.75",
                             "mobilenet_v2_0.50", "mobilenet_v2_0.35", "se_resnet50", "se_resnet101", "se_resnet152",
                             "efficientnet-b0", "efficientnet-b1", "efficientnet-b2", "efficientnet-b3",
                             "efficientnet-b4", "efficientnet-b5", "efficientnet-b6", "efficientnet-b7"],
                    help='The name of model, select one to train.')
parser.add_argument('--learn_rate_init', type=float, default=1e-4,
                    help='Initial value of cosine annealing learning rate.')
parser.add_argument('--learn_rate_end', type=float, default=1e-5,
                    help='End value of cosine annealing learning rate.')
parser.add_argument('--first_stage_epochs', type=int, default=30,
                    help='The Freeze phase trains the number of epochs.')
parser.add_argument('--second_stage_epochs', type=int, default=50,
                    help='The Finetune phase trains the number of epochs.')

args = parser.parse_args()

preprocess_dict = {
    "alexnet": "normal",
    "googlenet": "normal",
    "vgg": "caffe",
    "inception_v3": "tf",
    "xception": "tf",
    "resnet": "caffe",
    "se_resnet": "caffe",
    "densenet": "torch",
    "mobilenet": "tf",
    "efficientnet": None,
}
