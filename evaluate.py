# -*- coding: utf-8 -*-
# @File : evaluate.py
# @Author: Runist
# @Time : 2021/5/19 12:13
# @Software: PyCharm
# @Brief: 测试性能指标
import os
import numpy as np
from tqdm import tqdm
import tensorflow as tf
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score

from core.config import args
from core.dataset import Dataset
from core.models import get_model


def evaluate(model, val_file_path, label_name):
    """

    :param model: 模型对象
    :param val_file_path: 验证集文件路径
    :param label_name: 分类的名字
    :return: None
    """
    test_dataset = Dataset(val_file_path, label_name, batch_size=1)
    test_gen = test_dataset.tf_dataset()

    y_true = []
    y_pred = []

    process_bar = tqdm(test_gen.take(len(test_dataset)), ncols=80, unit="step")

    for image, label in process_bar:
        pred = model(image)

        pred = np.argmax(pred, axis=-1).astype(np.uint8)
        label = np.argmax(label, axis=-1).astype(np.uint8)

        y_pred.append(pred[0])
        y_true.append(label[0])

        accuracy = accuracy_score(np.array(y_true), np.array(y_pred))
        process_bar.set_postfix(accuracy=accuracy)

    cm = confusion_matrix(np.array(y_true), np.array(y_pred))
    accuracy = accuracy_score(np.array(y_true), np.array(y_pred))
    precision = precision_score(np.array(y_true), np.array(y_pred), average='macro')
    recall = recall_score(np.array(y_true), np.array(y_pred), average='macro')

    print("accuracy = {:.4f}, precision = {:.4f}, recall = {:.4f}".format(accuracy, precision, recall))
    print("Confusion matrix: \n", cm)


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    gpus = tf.config.experimental.list_physical_devices("GPU")
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)

    model = get_model(args.network, args.input_shape, args.num_classes)
    model.load_weights("./weights/{}/epoch=99_val_loss=0.1795_val_acc=0.9625.h5".format(args.network))
    # 如果载入的unfreeze之前的权重，则需要将trainable置为false
    model.trainable = False
    model.compile()

    evaluate(model, args.test_image_dir, args.label_name)
