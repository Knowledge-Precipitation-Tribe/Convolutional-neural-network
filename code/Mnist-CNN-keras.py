# -*- coding: utf-8 -*-#
'''
# Name:         Mnist-CNN-keras
# Description:  
# Author:       super
# Date:         2020/7/3
'''

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