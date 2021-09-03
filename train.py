# -*- coding: utf-8 -*-
# @File : train.py
# @Author: Runist
# @Time : 2021/7/2 15:04
# @Software: PyCharm
# @Brief: 训练脚本

import os
import tensorflow as tf
from tensorflow.keras import callbacks, optimizers, losses
from core.config import args
from core.dataset import Dataset
from core.models import get_model
from core.callback import CosineAnnealingLRScheduler, print_lr


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    gpus = tf.config.experimental.list_physical_devices("GPU")
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)

    if not os.path.exists("./weights/{}".format(args.network)):
        os.makedirs("./weights/{}".format(args.network))

    print("Preparing train {}.".format(args.network))
    model = get_model(args.network, args.input_shape, args.num_classes, include_top=args.include_top)

    cbk = [
        callbacks.ModelCheckpoint(
            './weights/{}/'.format(args.network)
            + 'epoch={epoch:02d}_val_loss={val_loss:.04f}_val_acc={val_accuracy:.04f}.h5',
            save_weights_only=True)
    ]

    # 第一阶段
    if args.first_stage_epochs > 0:

        train_dataset = Dataset(args.train_image_dir, args.label_name, batch_size=args.batch_size, aug=True)
        test_dataset = Dataset(args.test_image_dir, args.label_name, batch_size=1)

        train_steps = len(train_dataset) // args.batch_size
        test_steps = len(test_dataset)

        train_gen = train_dataset.tf_dataset()
        test_gen = test_dataset.tf_dataset()

        learning_rate = CosineAnnealingLRScheduler(args.first_stage_epochs * train_steps,
                                                   args.learn_rate_init, args.learn_rate_end,
                                                   warmth_rate=0.05)
        optimizer = optimizers.Adam(learning_rate)

        if args.include_top:
            freeze_layers = len(model.layers) - 1
        else:
            freeze_layers = len(model.layers) - 4

        for i in range(freeze_layers):
            model.layers[i].trainable = False
        print('Freeze the first {} layers of total {} layers. Train {} epoch.'.format(freeze_layers,
                                                                                      len(model.layers),
                                                                                      args.first_stage_epochs))

        model.compile(optimizer=optimizer,
                      loss=losses.CategoricalCrossentropy(),
                      metrics=['accuracy'])
        model.fit(train_gen,
                  steps_per_epoch=train_steps,
                  validation_data=test_gen,
                  validation_steps=test_steps,
                  epochs=args.first_stage_epochs,
                  shuffle=True,
                  callbacks=cbk)

    # 第二阶段
    if args.second_stage_epochs > 0:

        train_dataset = Dataset(args.train_image_dir, args.label_name, batch_size=args.batch_size, aug=True)
        test_dataset = Dataset(args.test_image_dir, args.label_name, batch_size=1)

        train_steps = len(train_dataset) // args.batch_size
        test_steps = len(test_dataset)

        train_gen = train_dataset.tf_dataset()
        test_gen = test_dataset.tf_dataset()

        learning_rate = CosineAnnealingLRScheduler(args.second_stage_epochs * train_steps,
                                                   args.learn_rate_init/10, args.learn_rate_end/10,
                                                   warmth_rate=0.05)
        optimizer = optimizers.Adam(learning_rate)

        for i in range(len(model.layers)):
            if 'BatchNormalization' in str(model.layers[i].name_scope):
                continue
            model.layers[i].trainable = True

        print('Unfreeze all layers. Train {} epoch.'.format(args.second_stage_epochs))

        model.compile(optimizer=optimizer,
                      loss=losses.CategoricalCrossentropy(),
                      metrics=['accuracy'])
        model.fit(train_gen,
                  steps_per_epoch=train_steps,
                  validation_data=test_gen,
                  validation_steps=test_steps,
                  epochs=args.first_stage_epochs + args.second_stage_epochs,
                  initial_epoch=args.first_stage_epochs,
                  shuffle=True,
                  callbacks=cbk)
