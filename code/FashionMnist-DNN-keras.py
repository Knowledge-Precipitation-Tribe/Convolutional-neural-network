# -*- coding: utf-8 -*-#
'''
# Name:         FashionMnist-DNN-keras
# Description:  
# Author:       super
# Date:         2020/7/3
'''

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