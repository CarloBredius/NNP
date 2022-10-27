#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Code from Mateus Espadoto (2019)

import pandas as pd
from glob import glob
from keras import applications
from keras import datasets as kdatasets
from skimage import io, transform
from keras.preprocessing.image import ImageDataGenerator
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
import numpy as np
import os
import scipy.io as sio
from scipy.io import arff
import tarfile
import tempfile
import zipfile
import pickle
import pandas as pd
import wget

def load_mat(file):
    data = sio.loadmat(file)

    X = np.rollaxis(data['X'], 3, 0)
    y = data['y'].squeeze()

    return X, y

def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo)
    return dict

def dont_process_reuters():
    if os.path.exists('data/X_reuters.npy'):
        return

    (x_train, y_train), (_, _) = kdatasets.reuters.load_data(skip_top=0, test_split=0.0, seed=420)
    word_index = kdatasets.reuters.get_word_index()

    reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])
    sentences = {}

    # Choose which classes of the dataset will be used
    classes = [1, 3, 4, 8, 10, 11, 13, 16, 19, 20, 21]

    for c in classes:
        print(c, np.sum(y_train == c))
        x_sentences = x_train[np.where(y_train == c)]

        sentences[c] = []

        for x in x_sentences:
            decoded_newswire = ' '.join([reverse_word_index.get(i - 3, '?') for i in x])
            sentences[c].append(decoded_newswire)

    reuters_sentences = []
    reuters_labels = []

    for c in classes:
        reuters_sentences += sentences[c]
        reuters_labels += list(np.repeat(c, len(sentences[c])))

    tfidf = TfidfVectorizer(max_features=5000)
    lenc = LabelEncoder()

    X_reuters = tfidf.fit_transform(reuters_sentences)
    X_reuters = X_reuters.todense()
    y_reuters = lenc.fit_transform(reuters_labels)
    np.save('data/X_reuters.npy', X_reuters)
    np.save('data/y_reuters.npy', y_reuters)

def dont_process_cifar10():
    if os.path.exists('data/X_cifar10_img.npy'):
        return

    (X_test, y_test), (X_train, y_train) = kdatasets.cifar10.load_data()

    y_test = y_test.squeeze()
    y_train = y_train.squeeze()

    X = np.vstack((X_test, X_train))
    y = np.hstack((y_test, y_train))
    X = X / 255.0
    
    model = applications.densenet.DenseNet201(include_top=False, weights='imagenet', input_shape=(32, 32, 3), pooling='max')
    X_new = model.predict(X, verbose=1, batch_size=512)
    model = None

    np.save('data/X_cifar10_img.npy', X)
    np.save('data/X_cifar10_densenet.npy', X_new)
    np.save('data/y_cifar10.npy', y)

def process_mnist():
    if os.path.exists('data/X_mnist.npy'):
        return

    (X_test, y_test), (X_train, y_train) = kdatasets.mnist.load_data()

    X = np.vstack((X_test, X_train))
    y = np.hstack((y_test, y_train))
    X = X.reshape((-1, 28 * 28)) / 255.0
    y = y.squeeze()

    np.save('data/X_mnist.npy', X)
    np.save('data/y_mnist.npy', y)


def dont_process_fashion():
    if os.path.exists('data/X_fashion.npy'):
        return

    (X_test, y_test), (X_train, y_train) = kdatasets.fashion_mnist.load_data()

    X = np.vstack((X_test, X_train))
    y = np.hstack((y_test, y_train))
    X = X.reshape((-1, 28 * 28)) / 255.0
    y = y.squeeze()

    np.save('data/X_fashion.npy', X)
    np.save('data/y_fashion.npy', y)

if __name__ == '__main__':
    if not os.path.exists('data'):
        os.mkdir('data')

    for func in sorted([f for f in dir() if f[:8] == 'process_']):
        print(str(func))
        globals()[func]()





