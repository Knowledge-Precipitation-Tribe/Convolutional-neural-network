# -*- coding: utf-8 -*-#
'''
# Name:         ColorAndShapeClassification-DNN-keras
# Description:  
# Author:       super
# Date:         2020/7/3
'''

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