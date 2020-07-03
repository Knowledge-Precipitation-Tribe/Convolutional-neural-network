# -*- coding: utf-8 -*-#
'''
# Name:         Cifar10-Resnet50-keras
# Description:  
# Author:       super
# Date:         2020/7/3
'''

import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import pickle

from sklearn.preprocessing import MinMaxScaler

from keras.utils import to_categorical
from keras.models import Sequential, load_model
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Input, Activation, add, BatchNormalization, AveragePooling2D
from keras.applications import ResNet50
from keras.regularizers import l2

names = ["airplane", "car", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]

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


def process_train_data(x_train, y_train):
    minmax = MinMaxScaler()
    x_train_rows = x_train.reshape(x_train.shape[0], 32 * 32 * 3)
    x_train = minmax.fit_transform(x_train_rows)
    x_train = x_train.reshape(x_train.shape[0], 32, 32, 3)

    y_train = to_categorical(y_train)
    return x_train, y_train


def load_test_data(data_path):
    batch = unpickle(data_path + '/test_batch')
    x_test = batch[b'data'].reshape((len(batch[b'data']), 3, 32, 32)).transpose(0, 2, 3, 1)
    y_test = batch[b'labels']
    return x_test, y_test

def process_test_data(x_test, y_test):
    minmax = MinMaxScaler()
    x_test_rows = x_test.reshape(x_test.shape[0], 32 * 32 * 3)
    x_test = minmax.fit_transform(x_test_rows)
    x_test = x_test.reshape(x_test.shape[0], 32, 32, 3)

    y_test = to_categorical(y_test)
    return x_test, y_test


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

def show_result(x, y, y_raw):
    fig, ax = plt.subplots(nrows=8, ncols=8, figsize=(11, 11))
    for i in range(64):
        ax[i // 8, i % 8].imshow(x[i])
        if y[i] == y_raw[i]:
            ax[i // 8, i % 8].set_title(names[y_raw[i]])
        else:
            ax[i // 8, i % 8].set_title(names[y_raw[i]]+ "(" + names[y[i]] + ")", fontdict={'color':'r'})
        ax[i // 8, i % 8].axis('off')
    # endfor
    plt.show()


def build_model():
    model = ResNet50(weights=None, classes=10, input_shape=(32,32,3))
    model.compile(optimizer='rmsprop',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model


if __name__ == "__main__":
    x_train, y_train = load_train_data('../data/cifar')
    x_train, y_train = process_train_data(x_train, y_train)

    x_test, y_test = load_test_data('../data/cifar')
    x_test, y_test = process_train_data(x_test, y_test)
    print(x_train.shape)
    print(y_train.shape)
    print(x_test.shape)
    print(y_test.shape)

    model_path = "cifar/cifar_model_resnet50.h5"


    if os.path.exists(model_path):
        model = load_model(model_path)
    else:
        model = build_model()
        history = model.fit(x_train, y_train,
                            batch_size=64,
                            epochs=1,
                            validation_split=0.2)
        model.save(model_path)
        draw_train_history(history)

    loss, accuracy = model.evaluate(x_test, y_test)
    print("test loss: {}, test accuracy: {}".format(loss, accuracy))

    z = model.predict(x_test[0:64])
    show_result(x_test[0:64], np.argmax(z, axis=1), np.argmax(y_test, axis=1))