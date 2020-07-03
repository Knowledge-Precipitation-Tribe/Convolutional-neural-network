# Cifar-10分类-ResNet-v2

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
from keras.models import Model, load_model
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Input, Activation, add, BatchNormalization, AveragePooling2D
from keras_applications import resnet
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


def resnet_layer(inputs,
                 num_filters=16,
                 kernel_size=3,
                 strides=1,
                 activation='relu',
                 batch_normalization=True,
                 conv_first=True):
    """2D 卷积批量标准化 - 激活栈构建器

    # 参数
        inputs (tensor): 从输入图像或前一层来的输入张量
        num_filters (int): Conv2D 过滤器数量
        kernel_size (int): Conv2D 方形核维度
        strides (int): Conv2D 方形步幅维度
        activation (string): 激活函数名
        batch_normalization (bool): 是否包含批标准化
        conv_first (bool): conv-bn-activation (True) 或
            bn-activation-conv (False)

    # 返回
        x (tensor): 作为下一层输入的张量
    """
    conv = Conv2D(num_filters,
                  kernel_size=kernel_size,
                  strides=strides,
                  padding='same',
                  kernel_initializer='he_normal',
                  kernel_regularizer=l2(1e-4))

    x = inputs
    if conv_first:
        x = conv(x)
        if batch_normalization:
            x = BatchNormalization()(x)
        if activation is not None:
            x = Activation(activation)(x)
    else:
        if batch_normalization:
            x = BatchNormalization()(x)
        if activation is not None:
            x = Activation(activation)(x)
        x = conv(x)
    return x

def build_model(input_shape, depth, num_classes=10):
    """ResNet 版本 2 模型构建器 [b]

    (1 x 1)-(3 x 3)-(1 x 1) BN-ReLU-Conv2D 的堆栈，也被称为瓶颈层。
    每一层的第一个快捷连接是一个 1 x 1 Conv2D。
    第二个及以后的快捷连接是 identity。
    在每个阶段的开始，特征图大小由具有 strides=2 的卷积层减半（下采样），
    而滤波器的数量加倍。在每个阶段中，这些层具有相同数量的过滤器和相同的特征图尺寸。
    特征图尺寸:
    conv1  : 32x32,  16
    stage 0: 32x32,  64
    stage 1: 16x16, 128
    stage 2:  8x8,  256

    # 参数
        input_shape (tensor): 输入图像张量的尺寸
        depth (int): 核心卷积层的数量
        num_classes (int): 类别数 (CIFAR10 为 10)

    # 返回
        model (Model): Keras 模型实例
    """
    if (depth - 2) % 9 != 0:
        raise ValueError('depth should be 9n+2 (eg 56 or 110 in [b])')
    # 开始模型定义。
    num_filters_in = 16
    num_res_blocks = int((depth - 2) / 9)

    inputs = Input(shape=input_shape)
    # v2 在将输入分离为两个路径前执行带 BN-ReLU 的 Conv2D 操作。
    x = resnet_layer(inputs=inputs,
                     num_filters=num_filters_in,
                     conv_first=True)

    # 实例化残差单元的栈
    for stage in range(3):
        for res_block in range(num_res_blocks):
            activation = 'relu'
            batch_normalization = True
            strides = 1
            if stage == 0:
                num_filters_out = num_filters_in * 4
                if res_block == 0:  # first layer and first stage
                    activation = None
                    batch_normalization = False
            else:
                num_filters_out = num_filters_in * 2
                if res_block == 0:  # first layer but not first stage
                    strides = 2    # downsample

            # 瓶颈残差单元
            y = resnet_layer(inputs=x,
                             num_filters=num_filters_in,
                             kernel_size=1,
                             strides=strides,
                             activation=activation,
                             batch_normalization=batch_normalization,
                             conv_first=False)
            y = resnet_layer(inputs=y,
                             num_filters=num_filters_in,
                             conv_first=False)
            y = resnet_layer(inputs=y,
                             num_filters=num_filters_out,
                             kernel_size=1,
                             conv_first=False)
            if res_block == 0:
                # 线性投影残差快捷键连接，以匹配更改的 dims
                x = resnet_layer(inputs=x,
                                 num_filters=num_filters_out,
                                 kernel_size=1,
                                 strides=strides,
                                 activation=None,
                                 batch_normalization=False)
            x = add([x, y])

        num_filters_in = num_filters_out

    # 在顶层添加分类器
    # v2 has BN-ReLU before Pooling
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = AveragePooling2D(pool_size=8)(x)
    y = Flatten()(x)
    outputs = Dense(num_classes,
                    activation='softmax',
                    kernel_initializer='he_normal')(y)

    # 实例化模型。
    model = Model(inputs=inputs, outputs=outputs)
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

    model_path = "cifar/cifar_model_resnet_v2.h5"

    depth = 3 * 9 + 2
    input_shape = (32,32,3)

    if os.path.exists(model_path):
        model = load_model(model_path)
    else:
        model = build_model(input_shape=input_shape, depth=depth)
        history = model.fit(x_train, y_train,
                            batch_size=64,
                            epochs=10,
                            validation_split=0.2)
        model.save(model_path)
        draw_train_history(history)

    loss, accuracy = model.evaluate(x_test, y_test)
    print("test loss: {}, test accuracy: {}".format(loss, accuracy))

    z = model.predict(x_test[0:64])
    show_result(x_test[0:64], np.argmax(z, axis=1), np.argmax(y_test, axis=1))
```

### 模型结构

![](../.gitbook/assets/image%20%2881%29.png)

### 模型输出

```python
test loss: 1.5088062532424926, test accuracy: 0.7302
```

### 模型损失以及准确率曲线

可以看到虽然只训练了二十个epoch，但是模型后面已经过拟合了。

![](../.gitbook/assets/image%20%28105%29.png)

### 分类结果

![](../.gitbook/assets/image%20%28101%29.png)

## 代码位置

\*\*\*\*[**Cifar10-ResNet-v2-keras**](https://github.com/Knowledge-Precipitation-Tribe/Convolutional-neural-network/blob/master/code/Cifar10-ResNet-v2-keras.py)\*\*\*\*

## 参考资料

\[1\] 参考 [https://keras-zh.readthedocs.io/examples/cifar10\_resnet/](https://keras-zh.readthedocs.io/examples/cifar10_resnet/)

