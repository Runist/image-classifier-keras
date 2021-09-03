# Image Classification 

This is for beginners to be able to easily use image classification and design of the general code, with keras implementation, **It will be continuously updated!**

## Introduction

### Implemented Network

- [x] AlexNet
- [x] VGG
- [x] GoogleNet
- [x] ResNet
- [x] MobileNet
- [x] DenseNet
- [x] SENet
- [x] EfficientNet
- [x] InceptionV3
- [x] Xception
- [ ] ShuffeNet

You can choose any network to train, the specific configuration is in *./core/config,py*.

### Dataset

A dataset of five flower species.

### pretrain weights

For convenience, I have uploaded the ImageNet pre-training weights to release.

## Quick start

1. clone this repository

```shell
git clone https://github.com/Runist/image-classifier-keras.git
```
2. You need to install some dependency package.

```shell
cd image-classifier-keras
pip installl -r requirements.txt
```
3. Download the **flower dataset**.
```shell
wget https://github.com/Runist/image-classifier-keras/releases/download/v0.2/dataset.zip
unzip dataset.zip
```

4. Start train your model.

```shell
python train.py
```
You will get the following output on the screen:

```shell
Downloading data from https://github.com/Runist/image-classifier-keras/releases/download/v0.2/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5
98%[=============================>]

Preparing train resnet50.
Freeze the first 176 layers of total 177 layers. Train 50 epoch.
Epoch 1/50
  8/103 [=>............................] - ETA: 2:03 - loss: 1.9460 - accuracy: 0.1172 - lr: 1.7510e-06
```

5. You can run *evaluate.py* to watch model performance.

```shell
python evaluate.py
```

```shell
100%|███████████████████████| 364/364 [00:26<00:00, 13.74step/s, accuracy=0.951]
accuracy = 0.9505, precision = 0.9505, recall = 0.9516
Confusion matrix: 
 [[62  0  0  0  1]
 [ 4 85  0  0  0]
 [ 0  2 59  0  3]
 [ 0  0  0 68  1]
 [ 1  2  3  1 72]]
```

## Other

To be continue...