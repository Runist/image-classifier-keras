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
from tensorflow.keras import losses
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score

import core.config as cfg
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
        logits = model.predict(image)
        pred = tf.nn.softmax(logits, axis=-1)

        pred = np.argmax(pred, axis=-1).astype(np.uint8)
        label = np.argmax(label, axis=-1).astype(np.uint8)

        y_pred.append(pred[0])
        y_true.append(label[0])

        process_bar.set_postfix(result=[pred[0] == label][0])

    cm = confusion_matrix(np.array(y_true), np.array(y_pred))
    accuracy = accuracy_score(np.array(y_true), np.array(y_pred))
    precision = precision_score(np.array(y_true), np.array(y_pred), average='macro')
    recall = recall_score(np.array(y_true), np.array(y_pred), average='macro')

    print("accuracy = {:.4f}, precision = {:.4f}, recall = {:.4f}".format(accuracy, precision, recall))
    print("Confusion matrix: \n", cm)


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    gpus = tf.config.experimental.list_physical_devices("GPU")
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)

    dataset = Dataset(cfg.test_data_path, cfg.label_name, batch_size=1)
    gen = dataset.tf_dataset()

    model = get_model(cfg.network, cfg.input_shape, cfg.num_classes)
    model.load_weights("./weights/{}/epoch=40_val_loss=0.3655_val_acc=0.9148.h5".format(cfg.network))
    # 如果载入的unfreeze之前的权重，则需要将trainable置为false
    model.trainable = False
    model.compile(loss=losses.CategoricalCrossentropy(from_logits=True), metrics=['accuracy'])

    evaluate(model, cfg.test_data_path, cfg.label_name)
