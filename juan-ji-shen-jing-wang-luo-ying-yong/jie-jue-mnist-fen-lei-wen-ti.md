# 解决MNIST分类问题

## 模型搭建

我们可以用一个三层的神经网络解决MNIST问题，并得到了97.49%的准确率。当时使用的模型如图18-31。

![&#x56FE;18-31 &#x524D;&#x9988;&#x795E;&#x7ECF;&#x7F51;&#x7EDC;&#x6A21;&#x578B;&#x89E3;&#x51B3;MNIST&#x95EE;&#x9898;](../.gitbook/assets/image%20%2844%29.png)

这一节中，我们将学习如何使用卷积网络来解决MNIST问题。首先搭建模型如图18-32。

![&#x56FE;18-32 &#x5377;&#x79EF;&#x795E;&#x7ECF;&#x7F51;&#x7EDC;&#x6A21;&#x578B;&#x89E3;&#x51B3;MNIST&#x95EE;&#x9898;](../.gitbook/assets/image%20%2821%29.png)

表18-5展示了模型中各层的功能和参数。

表18-5 模型中各层的功能和参数

| Layer | 参数 | 输入 | 输出 | 参数个数 |
| :--- | :--- | :--- | :--- | :--- |
| 卷积层 | 8x5x5,s=1 | 1x28x28 | 8x24x24 | 200+8 |
| 激活层 | 2x2,s=2, max | 8x24x24 | 8x24x24 |  |
| 池化层 | Relu | 8x24x24 | 8x12x12 |  |
| 卷积层 | 16x5x5,s=1 | 8x12x12 | 16x8x8 | 400+16 |
| 激活层 | Relu | 16x8x8 | 16x8x8 |  |
| 池化层 | 2x2, s=2, max | 16x8x8 | 16x4x4 |  |
| 全连接层 | 256x32 | 256 | 32 | 8192+32 |
| 批归一化层 |  | 32 | 32 |  |
| 激活层 | Relu | 32 | 32 |  |
| 全连接层 | 32x10 | 32 | 10 | 320+10 |
| 分类层 | softmax,10 | 10 | 10 |  |

卷积核的大小如何选取呢？大部分卷积神经网络都会用1、3、5、7的方式递增，还要注意在做池化时，应该尽量让输入的矩阵尺寸是偶数，如果不是的话，应该在上一层卷积层加padding，使得卷积的输出结果矩阵的宽和高为偶数。

## 代码实现

```python
def model():
    num_output = 10
    dataReader = LoadData(num_output)

    max_epoch = 5
    batch_size = 128
    learning_rate = 0.1
    params = HyperParameters_4_2(
        learning_rate, max_epoch, batch_size,
        net_type=NetType.MultipleClassifier,
        init_method=InitialMethod.Xavier,
        optimizer_name=OptimizerName.Momentum)

    net = NeuralNet_4_2(params, "mnist_conv_test")
    
    c1 = ConvLayer((1,28,28), (8,5,5), (1,0), params)
    net.add_layer(c1, "c1")
    r1 = ActivationLayer(Relu())
    net.add_layer(r1, "relu1")
    p1 = PoolingLayer(c1.output_shape, (2,2), 2, PoolingTypes.MAX)
    net.add_layer(p1, "p1") 
  
    c2 = ConvLayer(p1.output_shape, (16,5,5), (1,0), params)
    net.add_layer(c2, "23")
    r2 = ActivationLayer(Relu())
    net.add_layer(r2, "relu2")
    p2 = PoolingLayer(c2.output_shape, (2,2), 2, PoolingTypes.MAX)
    net.add_layer(p2, "p2")  

    f3 = FcLayer_2_0(p2.output_size, 32, params)
    net.add_layer(f3, "f3")
    bn3 = BnLayer(f3.output_size)
    net.add_layer(bn3, "bn3")
    r3 = ActivationLayer(Relu())
    net.add_layer(r3, "relu3")

    f4 = FcLayer_2_0(f3.output_size, 10, params)
    net.add_layer(f4, "f2")
    s4 = ClassificationLayer(Softmax())
    net.add_layer(s4, "s4")

    net.train(dataReader, checkpoint=0.05, need_test=True)
    net.ShowLossHistory(XCoordinate.Iteration)
```

## 运行结果

训练5个epoch后的损失函数值和准确率的历史记录曲线如图18-33。

![&#x56FE;18-33 &#x8BAD;&#x7EC3;&#x8FC7;&#x7A0B;&#x4E2D;&#x635F;&#x5931;&#x51FD;&#x6570;&#x503C;&#x548C;&#x51C6;&#x786E;&#x5EA6;&#x7684;&#x53D8;&#x5316;](../.gitbook/assets/image%20%2815%29.png)

打印输出结果如下：

```python
...
epoch=4, total_iteration=2133
loss_train=0.054449, accuracy_train=0.984375
loss_valid=0.060550, accuracy_valid=0.982000
save parameters
time used: 513.3446323871613
testing...
0.9865
```

最后可以得到98.44%的准确率，比全连接网络要高1个百分点。如果想进一步提高准确率，可以尝试增加卷积层的能力，比如使用更多的卷积核来提取更多的特征。

## 可视化

### 第一组的卷积可视化

下图按行显示了以下内容：

1. 卷积核数值
2. 卷积核抽象
3. 卷积结果
4. 激活结果
5. 池化结果

![&#x56FE;18-34 &#x5377;&#x79EF;&#x7ED3;&#x679C;&#x53EF;&#x89C6;&#x5316;](../.gitbook/assets/image%20%2869%29.png)

卷积核是5x5的，一共8个卷积核，所以第一行直接展示了卷积核的数值图形化以后的结果，但是由于色块太大，不容易看清楚其具体的模式，那么第二行的模式是如何抽象出来的呢？

因为特征是未知的，所以卷积神经网络不可能学习出类似下面的两个矩阵中左侧矩阵的整齐的数值，而很可能是如同右侧的矩阵一样具有很多噪音，但是大致轮廓还是个左上到右下的三角形，只是一些局部点上有一些值的波动。

```python
2  2  1  1  0               2  0  1  1  0
2  1  1  0  0               2  1  1  2  0
1  1  0 -1 -2               0  1  0 -1 -2
1  0 -1 -2 -3               1 -1  1 -4 -3
0 -1 -2 -3 -4               0 -1 -2 -3 -2
```

如何“看”出一个大概符合某个规律的模板呢？对此，笔者的心得是：

1. 摘掉眼镜（或者眯起眼睛）看第一行的卷积核的明暗变化模式；
2. 也可以用图像处理的办法，把卷积核形成的5x5的点阵做一个模糊处理；
3. 结合第三行的卷积结果推想卷积核的行为。

由此可以得到表18-6的模式。

表18-6 卷积核的抽象模式

| 卷积核序号 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| 抽象模式 | 右斜 | 下 | 中心 | 竖中 | 左下 | 上 | 右 | 左上 |

这些模式实际上就是特征，是卷积网络自己学习出来的，每一个卷积核关注图像的一个特征，比如上部边缘、下部边缘、左下边缘、右下边缘等。这些特征的排列有什么顺序吗？没有。每一次重新训练后，特征可能会变成其它几种组合，顺序也会发生改变，这取决于初始化数值及样本顺序、批大小等等因素。

当然可以用更高级的图像处理算法，对5x5的图像进行模糊处理，再从中提取模式。

### 第二组的卷积可视化

图18-35是第二组的卷积、激活、池化层的输出结果。

![&#x56FE;18-35 &#x7B2C;&#x4E8C;&#x7EC4;&#x5377;&#x79EF;&#x6838;&#x3001;&#x6FC0;&#x6D3B;&#x3001;&#x6C60;&#x5316;&#x7684;&#x53EF;&#x89C6;&#x5316;](../.gitbook/assets/image%20%2872%29.png)

* Conv2：由于是在第一层的特征图上卷积后叠加的结果，所以基本不能按照原图理解，但也能大致看出是是一些轮廓抽取的作用；
* Relu2：能看出的是如果黑色区域多的话，说明基本没有激活值，此卷积核效果就没用；
* Pool2：池化后分化明显的特征图是比较有用的特征，比如3、6、12、15、16；信息太多或者太少的特征图，都用途偏小，比如1、7、10、11。

## keras实现

```python
from keras.models import Sequential
from keras.layers import Conv2D,MaxPool2D,Flatten,Dense, BatchNormalization

import matplotlib.pyplot as plt

import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

from ExtendedDataReader.MnistImageDataReader import *

train_x = '../data/train-images-10'
train_y = '../data/train-labels-10'
test_x = '../data/test-images-10'
test_y = '../data/test-labels-10'

def load_data():
    print("reading data...")
    dataReader = MnistImageDataReader(train_x, train_y, test_x, test_y, "image")
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
    model.add(Conv2D(filters=8, kernel_size=(5,5), activation='relu', input_shape=(28,28,1)))
    model.add(MaxPool2D(pool_size=(2,2), strides=2))
    model.add(Conv2D(filters=16, kernel_size=(5,5), activation='relu'))
    model.add(MaxPool2D(pool_size=(2,2), strides=2))
    model.add(Flatten())
    model.add(Dense(32, activation='relu'))
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
        if y[i, 0] == y_raw[i, 0]:
            ax[i // 8, i % 8].set_title(y[i, 0])
        else:
            ax[i // 8, i % 8].set_title(y[i, 0], fontdict={'color':'r'})
        ax[i // 8, i % 8].axis('off')
    # endfor
    plt.show()


if __name__ == '__main__':
    x_train, y_train, x_test, y_test, x_val, y_val, x_test_raw, y_test_raw = load_data()
    print(x_train.shape)
    print(x_test.shape)
    print(x_val.shape)

    model = build_model()
    print(model.summary())
    model.save('mnist_cnn/keras-model.h5')
    history = model.fit(x_train, y_train,
                        epochs=5,
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

### 模型结构

![](../.gitbook/assets/image%20%28112%29.png)

### 模型输出

```python
Model: "sequential_1"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
conv2d_1 (Conv2D)            (None, 24, 24, 8)         208       
_________________________________________________________________
max_pooling2d_1 (MaxPooling2 (None, 12, 12, 8)         0         
_________________________________________________________________
conv2d_2 (Conv2D)            (None, 8, 8, 16)          3216      
_________________________________________________________________
max_pooling2d_2 (MaxPooling2 (None, 4, 4, 16)          0         
_________________________________________________________________
flatten_1 (Flatten)          (None, 256)               0         
_________________________________________________________________
dense_1 (Dense)              (None, 32)                8224      
_________________________________________________________________
batch_normalization_1 (Batch (None, 32)                128       
_________________________________________________________________
dense_2 (Dense)              (None, 10)                330       
=================================================================
Total params: 12,106
Trainable params: 12,042
Non-trainable params: 64
_________________________________________________________________

test loss: 0.06329172231366392, test accuracy: 0.9783999919891357
```

### 训练损失以及准确率曲线

![](../.gitbook/assets/image%20%28109%29.png)

### 分类结果

![](../.gitbook/assets/image%20%2894%29.png)

## 代码位置

原代码位置：[**Level4\_MnistConvNet**](https://github.com/microsoft/ai-edu/blob/master/A-%E5%9F%BA%E7%A1%80%E6%95%99%E7%A8%8B/A2-%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9C%E5%9F%BA%E6%9C%AC%E5%8E%9F%E7%90%86%E7%AE%80%E6%98%8E%E6%95%99%E7%A8%8B/SourceCode/ch18-CNNModel/Level4_MnistConvNet.py)\*\*\*\*

个人代码：[**Mnist-CNN**](https://github.com/Knowledge-Precipitation-Tribe/Convolutional-neural-network/blob/master/code/Mnist-CNN.py)\*\*\*\*

## 参考资料

* [http://scs.ryerson.ca/~aharley/vis/conv/](http://scs.ryerson.ca/~aharley/vis/conv/)

读者可以在上面这个网站看到MNIST的可视化结果，用鼠标可以改变三维视图的视角。

