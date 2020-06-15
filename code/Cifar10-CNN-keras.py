# -*- coding: utf-8 -*-#
'''
# Name:         Cifar10-CNN
# Description:  
# Author:       super
# Date:         2020/6/15
'''

import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import pickle

from sklearn.preprocessing import MinMaxScaler

from keras.utils import to_categorical
from keras.models import Sequential, load_model
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

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


def draw_img(data):
    """
    显示前60张图片
    :param data:
    :return:
    """
    fig, axes = plt.subplots(nrows=3, ncols=20, sharex=True, sharey=True, figsize=(80, 12))
    imgs = data[:60]

    for image, row in zip([imgs[:20], imgs[20:40], imgs[40:60]], axes):
        for img, ax in zip(image, row):
            ax.imshow(img)
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)

    fig.tight_layout(pad=0.1)
    plt.show()


def process_train_data(x_train, y_train):
    minmax = MinMaxScaler()
    x_train_rows = x_train.reshape(x_train.shape[0], 32 * 32 * 3)
    x_train = minmax.fit_transform(x_train_rows)
    x_train = x_train.reshape(x_train.shape[0], 32, 32, 3)

    y_train = to_categorical(y_train)
    return x_train, y_train


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

def build_model():
    model = Sequential()
    model.add(Conv2D(filters=16, kernel_size=2, padding='same', activation='relu', input_shape=(32, 32, 3)))
    model.add(MaxPooling2D(pool_size=2))

    model.add(Conv2D(filters=32, kernel_size=2, padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=2))

    model.add(Conv2D(filters=64, kernel_size=2, padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=2))
    model.add(Dropout(0.3))

    model.add(Flatten())
    model.add(Dense(500, activation='relu'))
    model.add(Dropout(0.4))
    model.add(Dense(10, activation='softmax'))

    model.compile(optimizer = 'rmsprop', loss ='categorical_crossentropy',metrics=['accuracy'])
    return model

def load_test_data(data_path):
    batch = unpickle(data_path + '/test_batch' + str(1))
    print(batch)
    x_test = batch[b'data'].reshape((len(batch[b'data']), 3, 32, 32)).transpose(0, 2, 3, 1)
    return x_test


def load_model(path):
    model = load_model(path)
    return model


if __name__ == "__main__":
    x_train, y_train = load_train_data('data')
    x_train, y_train = process_train_data(x_train, y_train)
    model_path = "cifar/cifar_model_cnn.h5"
    if os.path.exists(model_path):
        model = load_model(model_path)
    else:
        model = build_model()
        history = model.fit(x_train, y_train, batch_size=64, epochs=10, validation_split=0.2)
        draw_train_history(history)