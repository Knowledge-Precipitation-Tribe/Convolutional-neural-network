# 实现几何图形及颜色分类

## 提出问题

在前两节我们学习了如何按颜色分类和按形状分类几何图形，现在我们自然地想到如果把颜色和图形结合起来，卷积神经网络能不能正确分类呢？

请看样本数据，如图18-26。

![&#x56FE;18-26 &#x6837;&#x672C;&#x6570;&#x636E;](../.gitbook/assets/image%20%2830%29.png)

一共有3种形状及3种颜色，如表18-4所示。

表18-4 样本数据分类和数量

|  | 红色 | 蓝色 | 绿色 |
| :--- | :--- | :--- | :--- |
| 圆形 | 600:100 | 600:100 | 600:100 |
| 矩形 | 600:100 | 600:100 | 600:100 |
| 三角形 | 600:100 | 600:100 | 600:100 |

表中列出了9种样本的训练集和测试集的样本数量比例，都是600:100，

## 用前馈神经网络解决问题

我们仍然先使用全连接网络来解决这个问题，搭建一个三层的网络如下：

```python
ef dnn_model():
    num_output = 9
    max_epoch = 50
    batch_size = 16
    learning_rate = 0.01
    params = HyperParameters_4_2(
        learning_rate, max_epoch, batch_size,
        net_type=NetType.MultipleClassifier,
        init_method=InitialMethod.MSRA,
        optimizer_name=OptimizerName.Momentum)

    net = NeuralNet_4_2(params, "color_shape_dnn")
    
    f1 = FcLayer_2_0(784, 128, params)
    net.add_layer(f1, "f1")
    r1 = ActivationLayer(Relu())
    net.add_layer(r1, "relu1")

    f2 = FcLayer_2_0(f1.output_size, 64, params)
    net.add_layer(f2, "f2")
    r2 = ActivationLayer(Relu())
    net.add_layer(r2, "relu2")
    
    f3 = FcLayer_2_0(f2.output_size, num_output, params)
    net.add_layer(f3, "f3")
    s3 = ClassificationLayer(Softmax())
    net.add_layer(s3, "s3")

    return net
```

样本数据为3x28x28的彩色图，所以我们要把它转换成灰度图，然后再展开成1x784的向量，第一层用128个神经元，第二层用64个神经元，输出层用9个神经元接Softmax分类函数。

训练50个epoch后可以得到如下如图18-27所示的训练结果。

![&#x56FE;18-27 &#x8BAD;&#x7EC3;&#x8FC7;&#x7A0B;&#x4E2D;&#x635F;&#x5931;&#x51FD;&#x6570;&#x503C;&#x548C;&#x51C6;&#x786E;&#x5EA6;&#x7684;&#x53D8;&#x5316;](../.gitbook/assets/image%20%2826%29.png)

```python
......
epoch=49, total_iteration=15199
loss_train=0.003370, accuracy_train=1.000000
loss_valid=0.510589, accuracy_valid=0.883333
time used: 25.34346342086792
testing...
0.9011111111111111
load parameters
0.8988888888888888
```

在测试集上得到的准确度是89%，这已经超出笔者的预期了，本来猜测准确度会小于80%。有兴趣的读者可以再精调一下这个前馈神经网络网络，看看是否可以得到更高的准确度。

图18-28是部分测试集中的测试样本的预测结果。

![&#x56FE;18-28 &#x6D4B;&#x8BD5;&#x7ED3;&#x679C;](../.gitbook/assets/image%20%2859%29.png)

绝大部分样本预测是正确的，但是第3行第2列的样本，应该是green-rect，被预测成green-circle；最后两行的两个green-tri也被预测错了形状，颜色并没有错。

## 用卷积神经网络解决问题

下面我们来看看卷积神经网络能不能完成这个工作。首先搭建网络模型如下：

```python
def cnn_model():
    num_output = 9
    max_epoch = 20
    batch_size = 16
    learning_rate = 0.1
    params = HyperParameters_4_2(
        learning_rate, max_epoch, batch_size,
        net_type=NetType.MultipleClassifier,
        init_method=InitialMethod.MSRA,
        optimizer_name=OptimizerName.SGD)

    net = NeuralNet_4_2(params, "shape_color_cnn")
    
    c1 = ConvLayer((3,28,28), (8,3,3), (1,1), params)
    net.add_layer(c1, "c1")
    r1 = ActivationLayer(Relu())
    net.add_layer(r1, "relu1")
    p1 = PoolingLayer(c1.output_shape, (2,2), 2, PoolingTypes.MAX)
    net.add_layer(p1, "p1") 

    c2 = ConvLayer(p1.output_shape, (16,3,3), (1,0), params)
    net.add_layer(c2, "c2")
    r2 = ActivationLayer(Relu())
    net.add_layer(r2, "relu2")
    p2 = PoolingLayer(c2.output_shape, (2,2), 2, PoolingTypes.MAX)
    net.add_layer(p2, "p2") 

    params.learning_rate = 0.1

    f3 = FcLayer_2_0(p2.output_size, 32, params)
    net.add_layer(f3, "f3")
    bn3 = BnLayer(f3.output_size)
    net.add_layer(bn3, "bn3")
    r3 = ActivationLayer(Relu())
    net.add_layer(r3, "relu3")
    
    f4 = FcLayer_2_0(f3.output_size, num_output, params)
    net.add_layer(f4, "f4")
    s4 = ClassificationLayer(Softmax())
    net.add_layer(s4, "s4")

    return net
```

经过20个epoch的训练后，我们得到的结果如图18-29。

![&#x56FE;18-29 &#x8BAD;&#x7EC3;&#x8FC7;&#x7A0B;&#x4E2D;&#x635F;&#x5931;&#x51FD;&#x6570;&#x503C;&#x548C;&#x51C6;&#x786E;&#x5EA6;&#x7684;&#x53D8;&#x5316;](../.gitbook/assets/image%20%2845%29.png)

以下是打印输出的最后几行：

```text
......
epoch=19, total_iteration=6079
loss_train=0.005184, accuracy_train=1.000000
loss_valid=0.118708, accuracy_valid=0.957407
time used: 131.77996039390564
testing...
0.97
load parameters
0.97
```

可以看到我们在测试集上得到了97%的准确度，比DNN模型要高出很多，这也证明了卷积神经网络在图像识别上的能力。

图18-30是部分测试集中的测试样本的预测结果。

![&#x56FE;18-30 &#x6D4B;&#x8BD5;&#x7ED3;&#x679C;](../.gitbook/assets/image%20%2855%29.png)

绝大部分样本预测是正确的，只有最后一行第4个样本，本来是green-triangle，被预测成green-circle。

## keras实现

### DNN

```python
from ExtendedDataReader.GeometryDataReader import *

from keras.models import Sequential
from keras.layers import Dense

import matplotlib.pyplot as plt

import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

train_data_name = "../data/ch17.train_shape_color.npz"
test_data_name = "../data/ch17.test_shape_color.npz"

name = ["red-circle", "red-rect", "red-tri", "green-circle", "green-rect", "green-tri", "blue-circle", "blue-rect",
        "blue-tri", ]

def load_data(mode):
    print("reading data...")
    dataReader = GeometryDataReader(train_data_name, test_data_name, mode)
    dataReader.ReadData()
    dataReader.NormalizeX()
    dataReader.NormalizeY(NetType.MultipleClassifier, base=0)
    dataReader.Shuffle()
    dataReader.GenerateValidationSet(k=10)
    return dataReader

def data_process(dataReader):
    x_train, y_train = dataReader.XTrain, dataReader.YTrain
    x_test, y_test = dataReader.XTest, dataReader.YTest
    x_val, y_val = dataReader.XDev, dataReader.YDev

    x_test_raw = dataReader.XTestRaw[0:64]
    y_test_raw = dataReader.YTestRaw[0:64]

    return x_train, y_train, x_test, y_test, x_val, y_val, x_test_raw, y_test_raw

def build_model():
    model = Sequential()
    model.add(Dense(128, activation='relu', input_shape=(784, )))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(9, activation='softmax'))
    model.compile(optimizer='Adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model

#画出训练过程中训练和验证的精度与损失
def draw_train_history(history):
    plt.figure(1)

    # summarize history for accuracy
    plt.subplot(211)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'])

    # summarize history for loss
    plt.subplot(212)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'])
    plt.show()


def show_result(x, y,y_raw):
    fig, ax = plt.subplots(nrows=6, ncols=6, figsize=(9, 9))
    for i in range(36):
        ax[i // 6, i % 6].imshow(x[i].transpose(1, 2, 0))
        if np.argmax(y[i]) == np.argmax(y_raw[i]):
            ax[i // 6, i % 6].set_title(name[np.argmax(y[i])])
        else:
            ax[i // 6, i % 6].set_title(name[np.argmax(y[i])], fontdict={'color':'r'})
        ax[i // 6, i % 6].axis('off')
    # endfor
    plt.show()


if __name__ == '__main__':
    dataReader = load_data('vector')
    x_train, y_train, x_test, y_test, x_val, y_val, x_test_raw, y_test_raw = data_process(dataReader)
    print(x_train.shape)
    print(x_test.shape)
    print(x_val.shape)

    model = build_model()
    history = model.fit(x_train, y_train,
                        epochs=20,
                        batch_size=64,
                        validation_data=(x_val, y_val))
    draw_train_history(history)

    loss, accuracy = model.evaluate(x_test, y_test)
    print("test loss: {}, test accuracy: {}".format(loss, accuracy))

    X_test, Y_test = dataReader.GetTestSet()
    Z = model.predict(X_test[0:36])
    X = dataReader.XTestRaw[0:36] / 255
    Y = Y_test[0:36]
    show_result(X, Z, Y)

    weights = model.get_weights()
    print("weights: ", weights)
```

#### 模型输出

```python
test loss: 0.4167499415369497, test accuracy: 0.8755555748939514
```

#### 训练损失以及准确率曲线

![](../.gitbook/assets/image%20%2884%29.png)

#### 分类结果

![](../.gitbook/assets/image%20%2893%29.png)

### CNN

```python
from ExtendedDataReader.GeometryDataReader import *

from keras.models import Sequential
from keras.layers import Conv2D,MaxPool2D,Flatten,Dense, BatchNormalization

import matplotlib.pyplot as plt

import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

train_data_name = "../data/ch17.train_shape_color.npz"
test_data_name = "../data/ch17.test_shape_color.npz"

name = ["red-circle", "red-rect", "red-tri", "green-circle", "green-rect", "green-tri", "blue-circle", "blue-rect",
        "blue-tri", ]

def load_data(mode):
    print("reading data...")
    dataReader = GeometryDataReader(train_data_name, test_data_name, mode)
    dataReader.ReadData()
    dataReader.NormalizeX()
    dataReader.NormalizeY(NetType.MultipleClassifier, base=0)
    dataReader.Shuffle()
    dataReader.GenerateValidationSet(k=10)
    x_train, y_train = dataReader.XTrain, dataReader.YTrain
    x_test, y_test = dataReader.XTest, dataReader.YTest
    x_val, y_val = dataReader.XDev, dataReader.YDev

    x_train = x_train.reshape(x_train.shape[0], 28, 28, 3)
    x_test = x_test.reshape(x_test.shape[0], 28, 28, 3)
    x_val = x_val.reshape(x_val.shape[0], 28, 28, 3)

    x_test_raw = dataReader.XTestRaw[0:64]
    y_test_raw = dataReader.YTestRaw[0:64]

    return x_train, y_train, x_test, y_test, x_val, y_val, x_test_raw, y_test_raw

def build_model():
    model = Sequential()
    model.add(Conv2D(filters=8, kernel_size=(3,3), activation='relu', input_shape=(28,28,3)))
    model.add(MaxPool2D(pool_size=(2,2), strides=2))
    model.add(Conv2D(filters=16, kernel_size=(3,3), activation='relu'))
    model.add(MaxPool2D(pool_size=(2,2), strides=2))
    model.add(Flatten())
    model.add(Dense(32, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dense(9, activation='softmax'))
    model.compile(optimizer='Adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model

#画出训练过程中训练和验证的精度与损失
def draw_train_history(history):
    plt.figure(1)

    # summarize history for accuracy
    plt.subplot(211)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'])

    # summarize history for loss
    plt.subplot(212)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'])
    plt.show()


def show_result(x, y, y_raw):
    x = x / 255
    fig, ax = plt.subplots(nrows=8, ncols=8, figsize=(11, 11))
    for i in range(64):
        ax[i // 8, i % 8].imshow(x[i].transpose(1, 2, 0))
        if y[i, 0] == y_raw[i, 0]:
            ax[i // 8, i % 8].set_title(name[y[i, 0]])
        else:
            ax[i // 8, i % 8].set_title(name[y[i, 0]], fontdict={'color':'r'})
        ax[i // 8, i % 8].axis('off')
    # endfor
    plt.show()


if __name__ == '__main__':
    x_train, y_train, x_test, y_test, x_val, y_val, x_test_raw, y_test_raw = load_data('image')
    print(x_train.shape)
    print(x_test.shape)
    print(x_val.shape)

    model = build_model()
    print(model.summary())
    model.save('color_shape_cnn/keras-model.h5')
    history = model.fit(x_train, y_train,
                        epochs=20,
                        batch_size=64,
                        validation_data=(x_val, y_val))
    draw_train_history(history)

    loss, accuracy = model.evaluate(x_test, y_test)
    print("test loss: {}, test accuracy: {}".format(loss, accuracy))

    z = model.predict(x_test[0:64])
    show_result(x_test_raw[0:64], np.argmax(z, axis=1).reshape(64, 1), y_test_raw[0:64])

    weights = model.get_weights()
    print("weights: ", weights)
```

#### 模型结构

![](../.gitbook/assets/image%20%2896%29.png)

#### 模型输出

```python
Model: "sequential_1"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
conv2d_1 (Conv2D)            (None, 26, 26, 8)         224       
_________________________________________________________________
max_pooling2d_1 (MaxPooling2 (None, 13, 13, 8)         0         
_________________________________________________________________
conv2d_2 (Conv2D)            (None, 11, 11, 16)        1168      
_________________________________________________________________
max_pooling2d_2 (MaxPooling2 (None, 5, 5, 16)          0         
_________________________________________________________________
flatten_1 (Flatten)          (None, 400)               0         
_________________________________________________________________
dense_1 (Dense)              (None, 32)                12832     
_________________________________________________________________
batch_normalization_1 (Batch (None, 32)                128       
_________________________________________________________________
dense_2 (Dense)              (None, 9)                 297       
=================================================================
Total params: 14,649
Trainable params: 14,585
Non-trainable params: 64
_________________________________________________________________

test loss: 0.06439131122651613, test accuracy: 0.9800000190734863
```

#### 训练损失以及准确率曲线

![](../.gitbook/assets/image%20%2894%29.png)

#### 分类结果

![](../.gitbook/assets/image%20%2892%29.png)

## 代码位置

原代码位置：[ch18, Level3](https://github.com/microsoft/ai-edu/blob/master/A-%E5%9F%BA%E7%A1%80%E6%95%99%E7%A8%8B/A2-%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9C%E5%9F%BA%E6%9C%AC%E5%8E%9F%E7%90%86%E7%AE%80%E6%98%8E%E6%95%99%E7%A8%8B/SourceCode/ch18-CNNModel/Level3_ColorAndShape_DNN.py)

个人代码：

* \*\*\*\*[**ColorAndShapeClassification-DNN**](https://github.com/Knowledge-Precipitation-Tribe/Convolutional-neural-network/blob/master/code/ColorAndShapeClassification-DNN.py)\*\*\*\*
* \*\*\*\*[**ColorAndShapeClassification-CNN**](https://github.com/Knowledge-Precipitation-Tribe/Convolutional-neural-network/blob/master/code/ColorAndShapeClassification-CNN.py)\*\*\*\*

