# 实现颜色分类

## 提出问题

大家知道卷积神经网络可以在图像分类上发挥作用，而一般的图像都是彩色的，也就是说卷积神经网络应该可以判别颜色的。这一节中我们来测试一下颜色分类问题，也就是说，不管几何图形是什么样子的，只针对颜色进行分类。

先看一下样本数据，如图18-14。

![&#x56FE;18-14 &#x989C;&#x8272;&#x5206;&#x7C7B;&#x6837;&#x672C;&#x6570;&#x636E;&#x56FE;](../.gitbook/assets/image%20%287%29.png)

在样本数据中，一共有6种颜色，分别是：

* 红色 red
* 绿色 green
* 蓝色 blue
* 青色（蓝+绿） cyan
* 黄色（红+绿） yellow
* 粉色（红+蓝） pink

而这6种颜色是分布在5种形状之上的：

* 圆形
* 菱形
* 直线
* 矩形
* 三角形

我们看看神经网络能否排除形状的干扰，而单独把颜色区分开来。

## 用前馈神经网络解决问题

### 数据处理

由于输入图片是三通道的彩色图片，我们先把它转换成灰度图，

```python
class GeometryDataReader(DataReader_2_0):
    def ConvertToGray(self, data):
        (N,C,H,W) = data.shape
        new_data = np.empty((N,H*W))
        if C == 3: # color
            for i in range(N):
                new_data[i] = np.dot(
                    [0.299,0.587,0.114], 
                    data[i].reshape(3,-1)).reshape(1,784)
        elif C == 1: # gray
            new_data[i] = data[i,0].reshape(1,784)
        #end if
        return new_data
```

向量\[0.299,0.587,0.114\]的作用是，把三通道的彩色图片的RGB值与此向量相乘，得到灰度图，三个因子相加等于1，这样如果原来是\[255,255,255\]的话，最后的灰度图的值还是255。如果是\[255,255,0\]的话，最后的结果是：

$$
\begin{aligned}
Y &= 0.299 \cdot R + 0.586 \cdot G + 0.114 \cdot B \\
&= 0.299 \cdot 255 + 0.586 \cdot 255 + 0.114 \cdot 0 \\
&=225.675
\end{aligned} \tag{1}
$$

也就是说粉色的数值本来是\(255,255,0\)，变成了单一的值225.675。六种颜色中的每一种都会有不同的值，所以即使是在灰度图中，也会保留部分“彩色”信息，当然会丢失一些信息。这从公式1中很容易看出来，假设$$B=0$$，不同组合的$$R、G$$的值有可能得到相同的最终结果，因此会丢失彩色信息。

在转换成灰度图后，立刻用reshape\(1,784\)把它转变成矢量，该矢量就是每个样本的784维的特征值。

### 搭建模型

我们搭建的前馈神经网络模型如下：

```python
def dnn_model():
    num_output = 6
    max_epoch = 100
    batch_size = 16
    learning_rate = 0.01
    params = HyperParameters_4_2(
        learning_rate, max_epoch, batch_size,
        net_type=NetType.MultipleClassifier,
        init_method=InitialMethod.MSRA,
        optimizer_name=OptimizerName.SGD)

    net = NeuralNet_4_2(params, "color_dnn")
    
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

这就是一个普通的三层网络，两个隐层，神经元数量分别是128和64，一个输出层，最后接一个6分类Softmax。

### 运行结果

训练100个epoch后，得到如下损失函数图。

![&#x56FE;18-15 &#x8BAD;&#x7EC3;&#x8FC7;&#x7A0B;&#x4E2D;&#x7684;&#x635F;&#x5931;&#x51FD;&#x6570;&#x503C;&#x548C;&#x51C6;&#x786E;&#x5EA6;&#x53D8;&#x5316;&#x66F2;&#x7EBF;](../.gitbook/assets/image%20%2818%29.png)

从损失函数曲线可以看到，此网络已经有些轻微的过拟合了，如果重复多次运行训练过程，会得到75%到85%之间的一个准确度值，并不是非常稳定，但偏差也不会太大，这与样本的噪音有很大关系，比如一条很细的红色直线，可能会给训练带来一些不确定因素。

最后我们考察一下该模型在测试集上的表现：

```python
......
epoch=99, total_iteration=28199
loss_train=0.005832, accuracy_train=1.000000
loss_valid=0.593325, accuracy_valid=0.804000
save parameters
time used: 30.822062015533447
testing...
0.816
```

在图18-16的可视化结果，一共64张图，是测试集中1000个样本的前64个样本，每张图上方的标签是预测的结果。

![&#x56FE;18-16 &#x53EF;&#x89C6;&#x5316;&#x7ED3;&#x679C;](../.gitbook/assets/image%20%2831%29.png)

可以看到有很多直线的颜色被识别错了，比如最后一行的第1、3、5、6列，颜色错误。另外有一些大色块也没有识别对，比如第3行最后一列和第4行的头尾两个，都是大色块识别错误。也就是说，对两类形状上的颜色判断不准：

* 很细的线
* 很大的色块

这是什么原因呢？笔者分析：

1. 针对细直线，由于带颜色的像素点的数量非常少，被拆成向量后，这些像素点就会在1x784的矢量中彼此相距很远，特征不明显，很容易被判别成噪音；
2. 针对大色块，由于带颜色的像素点的数量非常多，即使被拆成向量，也会占据很大的部分，这样特征点与背景点的比例失衡，导致无法判断出到底哪个是特征点。

笔者认为以上两点是前馈神经网络在训练上的不稳定，以及最后准确度不高的主要原因。

当然有兴趣的读者也可以保留输入样本的三个彩色通道信息，把一个样本数据变成1x3x784=2352的向量进行试验，看看是不是可以提高准确率。

## 用卷积神经网络解决问题

下面我们看看卷积神经网络的表现。我们直接使用三通道的彩色图片，不需要再做数据转换了。

### 搭建模型

```python
def cnn_model():
    num_output = 6
    max_epoch = 20
    batch_size = 16
    learning_rate = 0.1
    params = HyperParameters_4_2(
        learning_rate, max_epoch, batch_size,
        net_type=NetType.MultipleClassifier,
        init_method=InitialMethod.MSRA,
        optimizer_name=OptimizerName.SGD)

    net = NeuralNet_4_2(params, "color_conv")
    
    c1 = ConvLayer((3,28,28), (2,1,1), (1,0), params)
    net.add_layer(c1, "c1")
    r1 = ActivationLayer(Relu())
    net.add_layer(r1, "relu1")
    p1 = PoolingLayer(c1.output_shape, (2,2), 2, PoolingTypes.MAX)
    net.add_layer(p1, "p1") 

    c2 = ConvLayer(p1.output_shape, (3,3,3), (1,0), params)
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

表18-1展示了在这个模型中各层的作用和参数。

表18-1 模型各层的参数

| ID | 类型 | 参数 | 输入尺寸 | 输出尺寸 |
| :--- | :--- | :--- | :--- | :--- |
| 1 | 卷积 | 2x1x1, S=1 | 3x28x28 | 2x28x28 |
| 2 | 激活 | Relu | 2x28x28 | 2x28x28 |
| 3 | 池化 | 2x2, S=2, Max | 2x14x14 | 2x14x14 |
| 4 | 卷积 | 3x3x3, S=1 | 2x14x14 | 3x12x12 |
| 5 | 激活 | Relu | 3x12x12 | 3x12x12 |
| 6 | 池化 | 2x2, S=2, Max | 3x12x12 | 3x6x6 |
| 7 | 全连接 | 32 | 108 | 32 |
| 8 | 归一化 |  | 32 | 32 |
| 9 | 激活 | Relu | 32 | 32 |
| 10 | 全连接 | 6 | 32 | 6 |
| 11 | 分类 | Softmax | 6 | 6 |

为什么第一梯队的卷积用2个卷积核，而第二梯队的卷积核用3个呢？只是经过调参试验的结果，是最小的配置。如果使用更多的卷积核当然可以完成问题，但是如果使用更少的卷积核，网络能力就不够了，不能收敛。

### 运行结果

经过20个epoch的训练后，得到的结果如图18-17。

![&#x56FE;18-17 &#x8BAD;&#x7EC3;&#x8FC7;&#x7A0B;&#x4E2D;&#x7684;&#x635F;&#x5931;&#x51FD;&#x6570;&#x503C;&#x548C;&#x51C6;&#x786E;&#x5EA6;&#x53D8;&#x5316;&#x66F2;&#x7EBF;](../.gitbook/assets/image%20%2856%29.png)

以下是打印输出的最后几行：

```python
......
epoch=19, total_iteration=5639
loss_train=0.005293, accuracy_train=1.000000
loss_valid=0.106723, accuracy_valid=0.968000
save parameters
time used: 17.295073986053467
testing...
0.963
```

可以看到我们在测试集上得到了96.3%的准确度，比前馈神经网络模型要高出很多，这也证明了卷积神经网络在图像识别上的能力。

图18-18是测试集中前64个测试样本的预测结果。

![&#x56FE;18-18 &#x6D4B;&#x8BD5;&#x7ED3;&#x679C;](../.gitbook/assets/image%20%285%29.png)

在这一批的样本中，只有左下角的一个绿色直线被预测成蓝色了，其它的没发生错误。

## 1x1卷积

读者可能还记得在GoogLeNet的Inception模块中，有1x1的卷积核。这初看起来是一个非常奇怪的做法，因为1x1的卷积核基本上失去了卷积的作用，并没有建立在同一个通道上的相邻像素之间的相关性。

在本例中，为了识别颜色，我们也使用了1x1的卷积核，并且能够完成颜色分类的任务，这是为什么呢？

我们以三通道的数据举例。

![&#x56FE;18-19 1x1&#x5377;&#x79EF;&#x6838;&#x7684;&#x5DE5;&#x4F5C;&#x539F;&#x7406;](../.gitbook/assets/image%20%2862%29.png)

假设有一个三通道的1x1的卷积核，其值为\[1,2,-1\]，则相当于把每个通道的同一位置的像素值乘以卷积核，然后把结果相加，作为输出通道的同一位置的像素值。以左上角的像素点为例：

$$
1 \times 1 + 1 \times 2 + 1 \times (-1)=2
$$

相当于把上图拆开成9个样本，其值为：

```python
[1,1,1] # 左上角点
[3,3,0] # 中上点
[0,0,0] # 右上角点
[2,0,0] # 左中点
[0,1,1] # 中点
[4,2,1] # 右中点
[1,1,1] # 左下角点
[2,1,1] # 下中点
[0,0,0] # 右下角点
```

上述值排成一个9行3列的矩阵，然后与一个3行1列的向量$$(1,2,-1)^T$$相乘，得到9行1列的向量，然后再转换成3x3的矩阵。当然在实际过程中，这个1x1的卷积核的数值是学习出来的，而不是人为指定的。

这样做可以达到两个目的：

1. 跨通道信息整合
2. 降维以减少学习参数

**所以1x1的卷积核关注的是不同通道的相同位置的像素之间的相关性**，而不是同一通道内的像素的相关性，在本例中，意味着它关心的彩色通道信息，通过不同的卷积核，把彩色通道信息转变成另外一种表达方式，在保留原始信息的同时，还实现了降维。

在本例中，第一层卷积如果使用3个卷积核，输出尺寸是3x28x28，和输入尺寸一样，达不到降维的作用。所以，一般情况下，会使用小于输入通道数的卷积核数量，比如输入通道为3，则使用2个或1个卷积核。在上例中，如果使用2个卷积核，则输出两张9x9的特征图，这样才能达到降维的目的。如果想升维，那么使用4个以上的卷积核就可以了。

## 颜色分类可视化解释

在这里笔者根据自己的理解，解释一下针对这个颜色分类问题，卷积神经网络是如何工作的。

![&#x56FE;18-20 &#x989C;&#x8272;&#x5206;&#x7C7B;&#x95EE;&#x9898;&#x7684;&#x53EF;&#x89C6;&#x5316;&#x89E3;&#x91CA;](../.gitbook/assets/image%20%2810%29.png)

如图18-20所示：

1. 第一行是原始彩色图片，三通道28x28，特意挑出来都是矩形的6种颜色。
2. 第二行是第一卷积组合梯队的第1个1x1的卷积核在原始图片上的卷积结果。由于是1x1的卷积核，相当于用3个浮点数分别乘以三通道的颜色值所得到和，只要是最后的值不一样就可以了，因为对于神经网络来说，没有颜色这个概念，只有数值。从人的角度来看，6张图的前景颜色是不同的（因为原始图的前景色是6种不同颜色）。
3. 第三行是第一卷积组合梯队的第2个1x1的卷积核在原始图片上的卷积结果。与2相似，只不过3个浮点数的数值不同而已，也是得到6张前景色不同的图。
4. 第四行是第二卷积组合梯队的三个卷积核的卷积结果图，把三个特征图当作RGB通道后所生成的彩色图。单独看三个特征图的话，人类是无法理解的，所以我们把三个通道变成假的彩色图，仍然可以做到6个样本不同色，但是出现了一些边框，可以认为是卷积层从颜色上抽取出的“特征”，也就是说卷积网络“看”到了我们人类不能理解的东西。
5. 第五行是第二卷积组合梯队的激活函数结果，和原始图片相差很大。

如果用人类的视觉神经系统做类比，两个1x1的卷积核可以理解为两只眼睛上的视网膜上的视觉神经细胞，把彩色信息转变成神经电信号传入大脑的过程。最后由全连接层做分类，相当于大脑中的视觉知识体系。

回到神经网络的问题上，只要ReLU的输出结果中仍然含有“颜色”信息（用假彩色图可以证明这一点），并且针对原始图像中的不同的颜色，会生成不同的假彩色图，最后的全连接网络就可以有分辨的能力。

举例来说，从图18-20看，第一行的红色到了第五行变成了黑色，绿色变成了淡绿色，等等，是一一对应的关系。如果红色和绿色都变成了黑色，那么将分辨不出区别来。

## keras实现

### DNN

```python
from ExtendedDataReader.GeometryDataReader import *

from keras.models import Sequential
from keras.layers import Dense

import matplotlib.pyplot as plt

import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

train_data_name = "../data/ch17.train_color.npz"
test_data_name = "../data/ch17.test_color.npz"

name = ["red", "green", "blue", "yellow", "cyan", "pink"]

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

    x_test_raw = dataReader.XTestRaw[0:64]
    y_test_raw = dataReader.YTestRaw[0:64]

    return x_train, y_train, x_test, y_test, x_val, y_val, x_test_raw, y_test_raw

def build_model():
    model = Sequential()
    model.add(Dense(128, activation='relu', input_shape=(784, )))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(6, activation='softmax'))
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
    x_train, y_train, x_test, y_test, x_val, y_val, x_test_raw, y_test_raw = load_data('vector')
    # print(x_train.shape)
    # print(x_test.shape)
    # print(x_val.shape)

    model = build_model()
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

#### 模型输出

```python
test loss: 0.9397198047637939, test accuracy: 0.7200000286102295
```

#### 训练损失以及准确率曲线

![](../.gitbook/assets/image%20%2889%29.png)

#### 模型分类结果

![](../.gitbook/assets/image%20%2883%29.png)

### CNN

```python
from ExtendedDataReader.GeometryDataReader import *

from keras.models import Sequential
from keras.layers import Conv2D,MaxPool2D,Flatten,Dense, BatchNormalization

import matplotlib.pyplot as plt

import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

train_data_name = "../data/ch17.train_color.npz"
test_data_name = "../data/ch17.test_color.npz"

name = ["red", "green", "blue", "yellow", "cyan", "pink"]

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
    model.add(Conv2D(filters=2, kernel_size=(1,1), activation='relu', input_shape=(28,28,3)))
    model.add(MaxPool2D(pool_size=(2,2)))
    model.add(Conv2D(filters=3, kernel_size=(3,3), activation='relu'))
    model.add(MaxPool2D(pool_size=(2,2), strides=2))
    model.add(Flatten())
    model.add(Dense(32, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dense(6, activation='softmax'))
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
    model.save('color_cnn/keras-model.h5')
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

![](../.gitbook/assets/image%20%2884%29.png)

#### 模型输出

```python
Model: "sequential_1"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
conv2d_1 (Conv2D)            (None, 28, 28, 2)         8         
_________________________________________________________________
max_pooling2d_1 (MaxPooling2 (None, 14, 14, 2)         0         
_________________________________________________________________
conv2d_2 (Conv2D)            (None, 12, 12, 3)         57        
_________________________________________________________________
max_pooling2d_2 (MaxPooling2 (None, 6, 6, 3)           0         
_________________________________________________________________
flatten_1 (Flatten)          (None, 108)               0         
_________________________________________________________________
dense_1 (Dense)              (None, 32)                3488      
_________________________________________________________________
batch_normalization_1 (Batch (None, 32)                128       
_________________________________________________________________
dense_2 (Dense)              (None, 6)                 198       
=================================================================
Total params: 3,879
Trainable params: 3,815
Non-trainable params: 64
_________________________________________________________________

test loss: 0.013688366492278874, test accuracy: 0.996999979019165
```

#### 训练损失以及准确率曲线

![](../.gitbook/assets/image%20%2881%29.png)

#### 模型分类结果

![](../.gitbook/assets/image%20%2886%29.png)

## 代码位置

原代码位置：[ch18, Level1](https://github.com/microsoft/ai-edu/blob/master/A-%E5%9F%BA%E7%A1%80%E6%95%99%E7%A8%8B/A2-%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9C%E5%9F%BA%E6%9C%AC%E5%8E%9F%E7%90%86%E7%AE%80%E6%98%8E%E6%95%99%E7%A8%8B/SourceCode/ch18-CNNModel/Level1_Color_DNN.py)

个人代码：

* \*\*\*\*[**ColorClassification-DNN**](https://github.com/microsoft/ai-edu/blob/master/A-%E5%9F%BA%E7%A1%80%E6%95%99%E7%A8%8B/A2-%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9C%E5%9F%BA%E6%9C%AC%E5%8E%9F%E7%90%86%E7%AE%80%E6%98%8E%E6%95%99%E7%A8%8B/SourceCode/ch18-CNNModel/Level1_Color_DNN.py)\*\*\*\*
* \*\*\*\*[**ColorClassification-CNN**](https://github.com/Knowledge-Precipitation-Tribe/Convolutional-neural-network/blob/master/code/ColorClassification-CNN.py)\*\*\*\*

