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
- [ ] EfficientNet
- [ ] Xception
- [ ] ShuffeNet

You can choose any network to train, the specific configuration is in *./core/config,py*.

### Dataset

A dataset of five flower species.

### pretrain weights

For convenience, I have uploaded the ImageNet pre-training weights to release.

## Quick start

1. clone this repository

```shell
git clone https://github.com/Runist/ImageClassifier-keras.git
```
2. You need to install some dependency package.

```shell
cd ImageClassifier-keras
pip installl -r requirements.txt
```
3. Download the **flower dataset**.
```shell
wget https://github.com/Runist/ImageClassifier-keras/releases/download/v0.1/dataset.zip
unzip dataset.zip
```

4. Start train your model.

```shell
python train.py
```
You will get the following output on the screen:

```shell
Downloading data from https://github.com/Runist/ImageClassifier-keras/releases/download/v0.1/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5
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
100%|███████████████████████| 364/364 [00:30<00:00, 11.82step/s, result=[ True]]
accuracy = 0.8489, precision = 0.8590, recall = 0.8459
Confusion matrix: 
 [[56  2  0  2  3]
 [ 8 80  0  0  1]
 [ 0  1 59  1  3]
 [18  0  2 42  7]
 [ 2  0  5  0 72]]
```

## Other

To be continue...