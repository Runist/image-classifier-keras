# -*- coding: utf-8 -*-
# @File : train.py
# @Author: Runist
# @Time : 2021/7/2 15:04
# @Software: PyCharm
# @Brief: 训练脚本

import os
import tensorflow as tf
from tensorflow.keras import callbacks, optimizers, losses
import core.config as cfg
from core.dataset import Dataset
from core.models import get_model
from core.callback import CosineAnnealingLRScheduler, print_lr


def train_by_fit(model, train_gen, test_gen, train_steps, test_steps):
    """
    fit方式训练
    :param model: 训练模型
    :param train_gen: 训练集生成器
    :param test_gen: 测试集生成器
    :param train_steps: 训练次数
    :param test_steps: 测试次数
    :return: None
    """

    cbk = [
        callbacks.ModelCheckpoint(
            './weights/{}/'.format(cfg.network)
            + 'epoch={epoch:02d}_val_loss={val_loss:.04f}_val_acc={val_accuracy:.04f}.h5',
            save_weights_only=True)
    ]

    learning_rate = CosineAnnealingLRScheduler(cfg.first_epochs,
                                               train_steps, cfg.lr_init, cfg.lr_end, warmth_rate=0.05)
    optimizer = optimizers.Adam(learning_rate)
    lr_info = print_lr(optimizer)

    for i in range(len(model.layers) - 1):
        # print(model.layers[i].name)
        model.layers[i].trainable = False
    print('Freeze the first {} layers of total {} layers. Train {} epoch.'.format(len(model.layers)-1,
                                                                                  len(model.layers),
                                                                                  cfg.first_epochs))

    model.compile(optimizer=optimizer,
                  loss=losses.CategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy', lr_info])
    model.fit(train_gen,
              steps_per_epoch=train_steps,
              validation_data=test_gen,
              validation_steps=test_steps,
              epochs=cfg.first_epochs,
              shuffle=True,
              callbacks=cbk)

    learning_rate = CosineAnnealingLRScheduler(cfg.second_epochs,
                                               train_steps, cfg.lr_init/10, cfg.lr_end/10, warmth_rate=0.05)
    optimizer = optimizers.Adam(learning_rate)
    lr_info = print_lr(optimizer)

    for i in range(len(model.layers)):
        model.layers[i].trainable = True

    print('Unfreeze all layers. Train {} epoch.'.format(cfg.second_epochs))

    model.compile(optimizer=optimizer,
                  loss=losses.CategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy', lr_info])
    model.fit(train_gen,
              steps_per_epoch=train_steps,
              validation_data=test_gen,
              validation_steps=test_steps,
              epochs=cfg.first_epochs + cfg.second_epochs,
              initial_epoch=cfg.first_epochs,
              shuffle=True,
              callbacks=cbk)


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    gpus = tf.config.experimental.list_physical_devices("GPU")
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)

    if not os.path.exists("./weights/{}".format(cfg.network)):
        os.makedirs("./weights/{}".format(cfg.network))

    train_dataset = Dataset(cfg.train_data_path, cfg.label_name, batch_size=cfg.batch_size, aug=True)
    test_dataset = Dataset(cfg.test_data_path, cfg.label_name, batch_size=1)

    train_steps = len(train_dataset) // cfg.batch_size
    test_steps = len(test_dataset)

    train_gen = train_dataset.tf_dataset()
    test_gen = test_dataset.tf_dataset()

    model = get_model(cfg.network, cfg.input_shape, cfg.num_classes, include_top=True)
    print("Preparing train {}.".format(cfg.network))
    train_by_fit(model, train_gen, test_gen, train_steps, test_steps)

    loss, acc, _ = model.evaluate(test_gen.take(test_steps))
    print(loss, acc)
