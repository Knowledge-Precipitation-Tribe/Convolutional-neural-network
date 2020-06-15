# -*- coding: utf-8 -*-#
'''
# Name:         FashionMnist-CNN
# Description:  
# Author:       super
# Date:         2020/6/15
'''

from MiniFramework.NeuralNet_4_2 import *
from ExtendedDataReader.MnistImageDataReader import *

train_x = '../data/FashionMnistTrainX'
train_y = '../data/FashionMnistTrainY'
test_x = '../data/FashionMnistTestX'
test_y = '../data/FashionMnistTestY'

# 0-T恤 1-裤子 2-套衫 3-连衣裙 4-外套 5-凉鞋 6-衬衫 7-运动鞋 8-包 9-短靴
names = ["T-shirt", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Ankle Boot"]


def LoadData(mode):
    mdr = MnistImageDataReader(train_x, train_y, test_x, test_y, mode)
    mdr.ReadData()
    mdr.NormalizeX()
    mdr.NormalizeY(NetType.MultipleClassifier, base=0)
    mdr.Shuffle()
    mdr.GenerateValidationSet(k=12)
    return mdr


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

    net = NeuralNet_4_2(params, "fashion_mnist_cnn")

    c1 = ConvLayer((1, 28, 28), (32, 3, 3), (1, 0), params)
    net.add_layer(c1, "c1")
    r1 = ActivationLayer(Relu())
    net.add_layer(r1, "relu1")
    p1 = PoolingLayer(c1.output_shape, (2, 2), 2, PoolingTypes.MAX)
    net.add_layer(p1, "p1")
    """
    c2 = ConvLayer(p1.output_shape, (16,3,3), (1,1), params)
    net.add_layer(c2, "23")
    r2 = ActivationLayer(Relu())
    net.add_layer(r2, "relu2")
    p2 = PoolingLayer(c2.output_shape, (2,2), 2, PoolingTypes.MAX)
    net.add_layer(p2, "p2")  
    """
    f3 = FcLayer_2_0(p1.output_size, 128, params)
    net.add_layer(f3, "f3")
    bn3 = BnLayer(f3.output_size)
    net.add_layer(bn3, "bn3")
    r3 = ActivationLayer(Relu())
    net.add_layer(r3, "relu3")

    f4 = FcLayer_2_0(f3.output_size, 10, params)
    net.add_layer(f4, "f4")
    s4 = ClassificationLayer(Softmax())
    net.add_layer(s4, "s4")

    return net


def show_result(x, y, z, title, mode):
    fig, ax = plt.subplots(nrows=6, ncols=6, figsize=(9, 10))
    for i in range(36):
        if mode == "vector":
            ax[i // 6, i % 6].imshow(x[i].reshape(28, 28), cmap='gray')
        else:
            ax[i // 6, i % 6].imshow(x[i, 0], cmap='gray')
        if y[i] == z[i]:
            ax[i // 6, i % 6].set_title(names[z[i]])
        else:
            ax[i // 6, i % 6].set_title("*" + names[z[i]] + "(" + str(y[i]) + ")")
        ax[i // 6, i % 6].axis('off')
    # endfor
    plt.suptitle(title)
    plt.show()


if __name__ == '__main__':
    mode = "image"
    dataReader = LoadData(mode)
    net = cnn_model()
    net.train(dataReader, checkpoint=0.1, need_test=True)
    net.ShowLossHistory(XCoordinate.Iteration)

    X_test, Y_test = dataReader.GetTestSet()
    count = 36
    X = X_test[0:count]
    Y = Y_test[0:count]
    Z = net.inference(X)
    show_result(X, np.argmax(Y, axis=1), np.argmax(Z, axis=1), "cnn predication", mode)