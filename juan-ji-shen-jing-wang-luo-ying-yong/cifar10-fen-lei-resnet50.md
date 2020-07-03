# Cifar-10分类-ResNet50

Cifar 是加拿大政府牵头投资的一个先进科学项目研究所。Hinton、Bengio和他的学生在2004年拿到了 Cifar 投资的少量资金，建立了神经计算和自适应感知项目。这个项目结集了不少计算机科学家、生物学家、电气工程师、神经科学家、物理学家、心理学家，加速推动了 Deep Learning 的进程。从这个阵容来看，DL 已经和 ML 系的数据挖掘分的很远了。Deep Learning 强调的是自适应感知和人工智能，是计算机与神经科学交叉；Data Mining 强调的是高速、大数据、统计数学分析，是计算机和数学的交叉。

Cifar-10 是由 Hinton 的学生 Alex Krizhevsky、Ilya Sutskever 收集的一个用于普适物体识别的数据集。

## 提出问题

我们在前面的学习中，使用了MNIST和Fashion-MNIST两个数据集来练习卷积网络的分类，但是这两个数据集都是单通道的灰度图。虽然我们用彩色的几何图形作为例子讲解了卷积网络的基本功能，但是仍然与现实的彩色世界有差距。所以，本节我们将使用Cifar-10数据集来进一步检验一下卷积神经网络的能力。

图18-41是Cifar-10的样本数据。

![&#x56FE;18-41 Cifar-10&#x6837;&#x672C;&#x6570;&#x636E;](../.gitbook/assets/image%20%2837%29.png)

1. airplane，飞机，6000张
2. automobile，汽车，6000张
3. bird，鸟，6000张
4. cat，猫，6000张
5. deer，鹿，6000张
6. dog，狗，6000张
7. frog，蛙，6000张
8. horse，马，6000张
9. ship，船，6000张
10. truck，卡车，6000张

Cifar-10 由60000张32\*32的 RGB 彩色图片构成，共10个分类。50000张训练，10000张测试。分为6个文件，5个训练数据文件，每个文件中包含10000张图片，随机打乱顺序，1个测试数据文件，也是10000张图片。这个数据集最大的特点在于将识别迁移到了普适物体，而且应用于多分类（姊妹数据集Cifar-100达到100类，ILSVRC比赛则是1000类）。

## 代码实现

```python
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import pickle

from sklearn.preprocessing import MinMaxScaler

from keras.utils import to_categorical
from keras.models import Sequential, load_model
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Input, Activation, add, BatchNormalization, AveragePooling2D
from keras.applications import ResNet50
from keras.regularizers import l2

names = ["airplane", "car", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]

def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict


def load_train_data(data_path):
    batch = unpickle(data_path + '/data_batch_' + str(1))
    x_train = batch[b'data'].reshape((len(batch[b'data']), 3, 32, 32)).transpose(0, 2, 3, 1)
    y_train = batch[b'labels']
    for i in range(2, 6):
        batch = unpickle(data_path + '/data_batch_' + str(i))
        x_train = np.concatenate([x_train, batch[b'data'].reshape((len(batch[b'data']), 3, 32, 32)).transpose(0, 2, 3, 1)])
        y_train = np.concatenate([y_train, batch[b'labels']])
    return x_train, y_train


def process_train_data(x_train, y_train):
    minmax = MinMaxScaler()
    x_train_rows = x_train.reshape(x_train.shape[0], 32 * 32 * 3)
    x_train = minmax.fit_transform(x_train_rows)
    x_train = x_train.reshape(x_train.shape[0], 32, 32, 3)

    y_train = to_categorical(y_train)
    return x_train, y_train


def load_test_data(data_path):
    batch = unpickle(data_path + '/test_batch')
    x_test = batch[b'data'].reshape((len(batch[b'data']), 3, 32, 32)).transpose(0, 2, 3, 1)
    y_test = batch[b'labels']
    return x_test, y_test

def process_test_data(x_test, y_test):
    minmax = MinMaxScaler()
    x_test_rows = x_test.reshape(x_test.shape[0], 32 * 32 * 3)
    x_test = minmax.fit_transform(x_test_rows)
    x_test = x_test.reshape(x_test.shape[0], 32, 32, 3)

    y_test = to_categorical(y_test)
    return x_test, y_test


def draw_train_history(history):
    plt.figure(1)

    # summarize history for accuracy
    plt.subplot(211)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')

    # summarize history for loss
    plt.subplot(212)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()

def show_result(x, y, y_raw):
    fig, ax = plt.subplots(nrows=8, ncols=8, figsize=(11, 11))
    for i in range(64):
        ax[i // 8, i % 8].imshow(x[i])
        if y[i] == y_raw[i]:
            ax[i // 8, i % 8].set_title(names[y_raw[i]])
        else:
            ax[i // 8, i % 8].set_title(names[y_raw[i]]+ "(" + names[y[i]] + ")", fontdict={'color':'r'})
        ax[i // 8, i % 8].axis('off')
    # endfor
    plt.show()


def build_model():
    model = ResNet50(weights=None, classes=10, input_shape=(32,32,3))
    model.compile(optimizer='rmsprop',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model


if __name__ == "__main__":
    x_train, y_train = load_train_data('../data/cifar')
    x_train, y_train = process_train_data(x_train, y_train)

    x_test, y_test = load_test_data('../data/cifar')
    x_test, y_test = process_train_data(x_test, y_test)
    print(x_train.shape)
    print(y_train.shape)
    print(x_test.shape)
    print(y_test.shape)

    model_path = "cifar/cifar_model_resnet50.h5"


    if os.path.exists(model_path):
        model = load_model(model_path)
    else:
        model = build_model()
        history = model.fit(x_train, y_train,
                            batch_size=64,
                            epochs=1,
                            validation_split=0.2)
        model.save(model_path)
        draw_train_history(history)

    loss, accuracy = model.evaluate(x_test, y_test)
    print("test loss: {}, test accuracy: {}".format(loss, accuracy))

    z = model.predict(x_test[0:64])
    show_result(x_test[0:64], np.argmax(z, axis=1), np.argmax(y_test, axis=1))
```

### 模型结构

![](../.gitbook/assets/image%20%2887%29.png)

### 模型输出

```python
test loss: 1.512419725227356, test accuracy: 0.6747
```

### 模型损失以及准确率曲线

![](../.gitbook/assets/image%20%28105%29.png)

### 分类结果

![](../.gitbook/assets/image%20%2897%29.png)

## 代码位置

\*\*\*\*[**Cifar10-Resnet50-keras**](https://github.com/Knowledge-Precipitation-Tribe/Convolutional-neural-network/blob/master/code/Cifar10-Resnet50-keras.py)\*\*\*\*

