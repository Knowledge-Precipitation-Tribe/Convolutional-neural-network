# -*- coding: utf-8 -*-#
'''
# Name:         ColorClassification-CNN
# Description:  
# Author:       super
# Date:         2020/6/15
'''

import time
import matplotlib.pyplot as plt

from MiniFramework.NeuralNet_4_2 import *
from ExtendedDataReader.GeometryDataReader import *

train_data_name = "../data/ch17.train_color.npz"
test_data_name = "../data/ch17.test_color.npz"

name = ["red", "green", "blue", "yellow", "cyan", "pink"]


def LoadData(mode):
    print("reading data...")
    dr = GeometryDataReader(train_data_name, test_data_name, mode)
    dr.ReadData()
    dr.NormalizeX()
    dr.NormalizeY(NetType.MultipleClassifier, base=0)
    dr.Shuffle()
    dr.GenerateValidationSet(k=10)
    return dr


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

    net = NeuralNet_4_2(params, "color_cnn")

    c1 = ConvLayer((3, 28, 28), (2, 1, 1), (1, 0), params)
    net.add_layer(c1, "c1")
    r1 = ActivationLayer(Relu())
    net.add_layer(r1, "relu1")
    p1 = PoolingLayer(c1.output_shape, (2, 2), 2, PoolingTypes.MAX)
    net.add_layer(p1, "p1")

    c2 = ConvLayer(p1.output_shape, (3, 3, 3), (1, 0), params)
    net.add_layer(c2, "c2")
    r2 = ActivationLayer(Relu())
    net.add_layer(r2, "relu2")
    p2 = PoolingLayer(c2.output_shape, (2, 2), 2, PoolingTypes.MAX)
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


def show_samples(x, y, title):
    x = x / 255
    fig, ax = plt.subplots(nrows=8, ncols=8, figsize=(11, 11))
    for i in range(64):
        ax[i // 8, i % 8].imshow(x[i].transpose(1, 2, 0))
        ax[i // 8, i % 8].set_title(name[y[i, 0]])
        ax[i // 8, i % 8].axis('off')
    # endfor
    plt.suptitle(title)
    plt.show()


def show_result(x, y, y_raw, title):
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
    plt.suptitle(title)
    plt.show()


def train_cnn():
    dataReader = LoadData('image')
    net = cnn_model()
    return net, dataReader


if __name__ == '__main__':
    net, dataReader = train_cnn()
    x, y = dataReader.XTrainRaw[0:64], dataReader.YTrainRaw[0:64]
    show_samples(x, y, "color:sample")
    net.train(dataReader, checkpoint=0.5, need_test=True)
    net.ShowLossHistory(XCoordinate.Iteration)

    X_test, Y_test = dataReader.GetTestSet()
    Z = net.inference(X_test[0:64])
    X = dataReader.XTestRaw[0:64]
    Y = dataReader.YTestRaw[0:64]
    show_result(X, np.argmax(Z, axis=1).reshape(64, 1),Y, "cnn:predication")