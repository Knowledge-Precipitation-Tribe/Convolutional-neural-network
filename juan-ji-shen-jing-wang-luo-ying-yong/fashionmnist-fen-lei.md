# Fashion-MNIST分类

## 提出问题

MNIST手写识别数据集，对卷积神经网络来说已经太简单了，于是科学家们增加了图片的复杂度，用10种物品代替了10个数字，图18-36是它们的部分样本。

![&#x56FE;18-36 &#x90E8;&#x5206;&#x6837;&#x672C;&#x56FE;&#x5C55;&#x793A;](../.gitbook/assets/image%20%2829%29.png)

每3行是一类样本，按样本类别（从0开始计数）分行显示：

1. T-Shirt，T恤衫（1-3行）
2. Trouser，裤子（4-6行）
3. Pullover，套头衫（7-9行）
4. Dress，连衣裙（10-12行）
5. Coat，外套（13-15行）
6. Sandal，凉鞋（16-18行）
7. Shirt，衬衫（19-21行）
8. Sneaker，运动鞋（22-24行）
9. Bag，包（25-27行）
10. Ankle Boot，短靴（28-30行）

## 用前馈神经网络来解决问题

### 搭建模型

```python
def dnn_model():
    num_output = 10
    max_epoch = 10
    batch_size = 128
    learning_rate = 0.1
    params = HyperParameters_4_2(
        learning_rate, max_epoch, batch_size,
        net_type=NetType.MultipleClassifier,
        init_method=InitialMethod.MSRA,
        optimizer_name=OptimizerName.Momentum)

    net = NeuralNet_4_2(params, "fashion_mnist_dnn")
    
    f1 = FcLayer_2_0(784, 128, params)
    net.add_layer(f1, "f1")
    bn1 = BnLayer(f1.output_size)
    net.add_layer(bn1, "bn1")
    r1 = ActivationLayer(Relu())
    net.add_layer(r1, "relu1")

    f2 = FcLayer_2_0(f1.output_size, 64, params)
    net.add_layer(f2, "f2")
    bn2 = BnLayer(f2.output_size)
    net.add_layer(bn2, "bn2")
    r2 = ActivationLayer(Relu())
    net.add_layer(r2, "relu2")
    
    f3 = FcLayer_2_0(f2.output_size, num_output, params)
    net.add_layer(f3, "f3")
    s3 = ClassificationLayer(Softmax())
    net.add_layer(s3, "s3")

    return net
```

### 训练结果

训练10个epoch后得到如图18-37所示曲线，可以看到网络能力已经接近极限了，再训练下去会出现过拟合现象，准确度也不一定能提高。

![&#x56FE;18-37 &#x8BAD;&#x7EC3;&#x8FC7;&#x7A0B;&#x4E2D;&#x635F;&#x5931;&#x51FD;&#x6570;&#x503C;&#x548C;&#x51C6;&#x786E;&#x5EA6;&#x7684;&#x53D8;&#x5316;](../.gitbook/assets/image%20%2851%29.png)

图18-38是在测试集上的预测结果。

![&#x56FE;18-38 &#x6D4B;&#x8BD5;&#x7ED3;&#x679C;](../.gitbook/assets/image%20%2817%29.png)

凡是类别名字前面带\*号的，表示预测错误，比如第3行第1列，本来应该是第7类“运动鞋”，却被预测成了“凉鞋”。

## 用卷积神经网络来解决问题

### 搭建模型

```python
def cnn_model():
    num_output = 10
    max_epoch = 10
    batch_size = 128
    learning_rate = 0.01
    params = HyperParameters_4_2(
        learning_rate, max_epoch, batch_size,
        net_type=NetType.MultipleClassifier,
        init_method=InitialMethod.Xavier,
        optimizer_name=OptimizerName.Momentum)

    net = NeuralNet_4_2(params, "fashion_mnist_conv_test")
    
    c1 = ConvLayer((1,28,28), (32,3,3), (1,0), params)
    net.add_layer(c1, "c1")
    r1 = ActivationLayer(Relu())
    net.add_layer(r1, "relu1")
    p1 = PoolingLayer(c1.output_shape, (2,2), 2, PoolingTypes.MAX)
    net.add_layer(p1, "p1") 

    f3 = FcLayer_2_0(p1.output_size, 128, params)
    net.add_layer(f3, "f3")
    r3 = ActivationLayer(Relu())
    net.add_layer(r3, "relu3")

    f4 = FcLayer_2_0(f3.output_size, 10, params)
    net.add_layer(f4, "f4")
    s4 = ClassificationLayer(Softmax())
    net.add_layer(s4, "s4")

    return net
```

此模型只有一层卷积层，使用了32个卷积核，尺寸为3x3，后接最大池化层，然后两个全连接层。

### 训练结果

训练10个epoch后得到如图18-39的曲线。

![&#x56FE;18-39 &#x8BAD;&#x7EC3;&#x8FC7;&#x7A0B;&#x4E2D;&#x635F;&#x5931;&#x51FD;&#x6570;&#x503C;&#x548C;&#x51C6;&#x786E;&#x5EA6;&#x7684;&#x53D8;&#x5316;](../.gitbook/assets/image%20%2871%29.png)

在测试集上得到91.12%的准确率，在测试集上的前几个样本的预测结果如图18-40所示。

![&#x56FE;18-40 &#x6D4B;&#x8BD5;&#x7ED3;&#x679C;](../.gitbook/assets/image%20%2842%29.png)

与前馈神经网络方案相比，这32个样本里只有一个错误，第4行最后一列，把第9类“短靴”预测成了“凉鞋”，因为这个样本中间有一个三角形的黑色块，与凉鞋的镂空设计很像。

## keras-DNN实现

```python
from ExtendedDataReader.MnistImageDataReader import *

from keras.models import Sequential
from keras.layers import Dense, BatchNormalization

import matplotlib.pyplot as plt

import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

train_x = '../data/FashionMnistTrainX'
train_y = '../data/FashionMnistTrainY'
test_x = '../data/FashionMnistTestX'
test_y = '../data/FashionMnistTestY'

# 0-T恤 1-裤子 2-套衫 3-连衣裙 4-外套 5-凉鞋 6-衬衫 7-运动鞋 8-包 9-短靴
names = ["T-shirt", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Ankle Boot"]


def load_data(mode):
    print("reading data...")
    dataReader = MnistImageDataReader(train_x, train_y, test_x, test_y, mode)
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

    x_train = x_train.reshape(-1, 784)
    x_test = x_test.reshape(-1, 784)
    x_val = x_val.reshape(-1, 784)

    x_test_raw = dataReader.XTestRaw[0:64]
    y_test_raw = dataReader.YTestRaw[0:64]

    return x_train, y_train, x_test, y_test, x_val, y_val, x_test_raw, y_test_raw

def build_model():
    model = Sequential()
    model.add(Dense(128, activation='relu', input_shape=(784, )))
    model.add(BatchNormalization())
    model.add(Dense(64, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dense(10, activation='softmax'))
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
        ax[i // 8, i % 8].imshow(x[i].transpose(1, 2, 0).squeeze())
        if y[i] == y_raw[i]:
            ax[i // 8, i % 8].set_title(names[y_raw[i]])
        else:
            ax[i // 8, i % 8].set_title(names[y_raw[i]] + "(" + names[y[i]] + ")", fontdict={'color':'r'})
        ax[i // 8, i % 8].axis('off')
    # endfor
    plt.show()


if __name__ == '__main__':
    dataReader = load_data("vector")
    x_train, y_train, x_test, y_test, x_val, y_val, x_test_raw, y_test_raw = data_process(dataReader)
    print(x_train.shape)
    print(x_test.shape)
    print(x_val.shape)

    model = build_model()
    history = model.fit(x_train, y_train,
                        epochs=5,
                        batch_size=64,
                        validation_data=(x_val, y_val))
    draw_train_history(history)

    loss, accuracy = model.evaluate(x_test, y_test)
    print("test loss: {}, test accuracy: {}".format(loss, accuracy))

    z = model.predict(x_test[0:64])
    show_result(x_test_raw[0:64], np.argmax(y_test, axis=1), np.argmax(z, axis=1))

    weights = model.get_weights()
    print("weights: ", weights)
```

### 模型输出

```python
test loss: 0.3737159375786781, test accuracy: 0.8648999929428101
```

### 训练损失以及准确率曲线

![](../.gitbook/assets/image%20%2897%29.png)

### 分类结果

![](../.gitbook/assets/image%20%2889%29.png)

## keras-CNN实现

```python
from keras.models import Sequential
from keras.layers import Conv2D,MaxPool2D,Flatten,Dense, BatchNormalization

import matplotlib.pyplot as plt

import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

from ExtendedDataReader.MnistImageDataReader import *

train_x = '../data/FashionMnistTrainX'
train_y = '../data/FashionMnistTrainY'
test_x = '../data/FashionMnistTestX'
test_y = '../data/FashionMnistTestY'

# 0-T恤 1-裤子 2-套衫 3-连衣裙 4-外套 5-凉鞋 6-衬衫 7-运动鞋 8-包 9-短靴
names = ["T-shirt", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Ankle Boot"]


def load_data(mode):
    print("reading data...")
    dataReader = MnistImageDataReader(train_x, train_y, test_x, test_y, mode)
    dataReader.ReadData()
    dataReader.NormalizeX()
    dataReader.NormalizeY(NetType.MultipleClassifier, base=0)
    dataReader.Shuffle()
    dataReader.GenerateValidationSet(k=12)
    x_train, y_train = dataReader.XTrain, dataReader.YTrain
    x_test, y_test = dataReader.XTest, dataReader.YTest
    x_val, y_val = dataReader.XDev, dataReader.YDev

    x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
    x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
    x_val = x_val.reshape(x_val.shape[0], 28, 28, 1)

    x_test_raw = dataReader.XTestRaw[0:64]
    y_test_raw = dataReader.YTestRaw[0:64]

    return x_train, y_train, x_test, y_test, x_val, y_val, x_test_raw, y_test_raw

def build_model():
    model = Sequential()
    model.add(Conv2D(filters=32, kernel_size=(3,3), activation='relu', input_shape=(28,28,1)))
    model.add(MaxPool2D(pool_size=(2,2), strides=2))
    model.add(Conv2D(filters=16, kernel_size=(3,3), activation='relu'))
    model.add(MaxPool2D(pool_size=(2,2), strides=2))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dense(10, activation='softmax'))
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
        ax[i // 8, i % 8].imshow(x[i].transpose(1, 2, 0).squeeze())
        if y[i] == y_raw[i]:
            ax[i // 8, i % 8].set_title(names[y_raw[i]])
        else:
            ax[i // 8, i % 8].set_title(names[y_raw[i]] + "(" + names[y[i]] + ")", fontdict={'color':'r'})
        ax[i // 8, i % 8].axis('off')
    # endfor
    plt.show()


if __name__ == '__main__':
    x_train, y_train, x_test, y_test, x_val, y_val, x_test_raw, y_test_raw = load_data("image")
    print(x_train.shape)
    print(x_test.shape)
    print(x_val.shape)

    model = build_model()
    print(model.summary())
    model.save('fashion_mnist_cnn/keras-model.h5')
    history = model.fit(x_train, y_train,
                        epochs=5,
                        batch_size=64,
                        validation_data=(x_val, y_val))
    draw_train_history(history)

    loss, accuracy = model.evaluate(x_test, y_test)
    print("test loss: {}, test accuracy: {}".format(loss, accuracy))

    z = model.predict(x_test[0:64])
    show_result(x_test_raw[0:64], np.argmax(y_test, axis=1), np.argmax(z, axis=1))

    weights = model.get_weights()
    print("weights: ", weights)
```

### 模型结构

![](../.gitbook/assets/image%20%28105%29.png)

### 模型输出

```python
Model: "sequential_1"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
conv2d_1 (Conv2D)            (None, 26, 26, 32)        320       
_________________________________________________________________
max_pooling2d_1 (MaxPooling2 (None, 13, 13, 32)        0         
_________________________________________________________________
conv2d_2 (Conv2D)            (None, 11, 11, 16)        4624      
_________________________________________________________________
max_pooling2d_2 (MaxPooling2 (None, 5, 5, 16)          0         
_________________________________________________________________
flatten_1 (Flatten)          (None, 400)               0         
_________________________________________________________________
dense_1 (Dense)              (None, 128)               51328     
_________________________________________________________________
batch_normalization_1 (Batch (None, 128)               512       
_________________________________________________________________
dense_2 (Dense)              (None, 10)                1290      
=================================================================
Total params: 58,074
Trainable params: 57,818
Non-trainable params: 256
_________________________________________________________________

test loss: 0.28777108699083326, test accuracy: 0.8909000158309937
```

### 训练损失以及准确率曲线

![](../.gitbook/assets/image%20%2881%29.png)

### 分类结果

![](../.gitbook/assets/image%20%2898%29.png)

## 代码位置

原代码位置：[ch18, Level5](https://github.com/microsoft/ai-edu/blob/master/A-%E5%9F%BA%E7%A1%80%E6%95%99%E7%A8%8B/A2-%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9C%E5%9F%BA%E6%9C%AC%E5%8E%9F%E7%90%86%E7%AE%80%E6%98%8E%E6%95%99%E7%A8%8B/SourceCode/ch18-CNNModel/Level5_FashionMnist_DNN.py)

个人代码：

* \*\*\*\*[**FashionMnist-DNN**](https://github.com/Knowledge-Precipitation-Tribe/Convolutional-neural-network/blob/master/code/FashionMnist-DNN.py)\*\*\*\*
* \*\*\*\*[**FashionMnist-CNN**](https://github.com/Knowledge-Precipitation-Tribe/Convolutional-neural-network/blob/master/code/FashionMnist-CNN.py)\*\*\*\*

