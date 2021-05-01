#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
from time import perf_counter

import joblib
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow import keras
from MulticoreTSNE import MulticoreTSNE as TSNE
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

import nnproj

def project(X, p):
    X_new = p.fit_transform(X)
    scaler = MinMaxScaler()
    return scaler.fit_transform(X_new)

cmap = plt.get_cmap('tab10')
print("Loading datasets...")
X_mnist = np.load('data/X_mnist.npy')
y_mnist = np.load('data/y_mnist.npy')
X_mnist_bin = X_mnist[np.isin(y_mnist, [0, 1])]
y_mnist_bin = y_mnist[np.isin(y_mnist, [0, 1])]
print("Loading datasets complete")

for label, X, y in zip(['mnist-bin', 'mnist-full'],
                            [X_mnist_bin, X_mnist],
                            [y_mnist_bin, y_mnist]):
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=10000, test_size=3000, random_state=420, stratify=y)
    train_size = 9000
    # A subset is taken to randomize the training a little?
    X_train_p, _, y_train_p, _ = train_test_split(X_train, y_train, train_size=train_size, random_state=420, stratify=y_train)

    model = keras.models.load_model("NNP_model_" + label)

    # Use NNP to project
    t0 = perf_counter()
    X_2d_pred = model.predict(X_test)
    infer_elapsed_time = perf_counter() - t0

    # Print infer time
    print("NNP infer time: " + str(infer_elapsed_time))

    # Create scatter-plot using NNP
    fig2, ax2 = plt.subplots(1, 1, figsize=(10, 10))
    fig2.tight_layout()
    for x, c in enumerate(np.unique(y_train_p)):
        ax2.axis('off')
        ax2.scatter(X_2d_pred[y_test==c,0],  X_2d_pred[y_test==c,1],  c=cmap(x), s=15, label=c, alpha=0.7)
    fig2.savefig("projection_" + label + ".png")
