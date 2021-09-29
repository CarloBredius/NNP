#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
from time import perf_counter

import joblib
import matplotlib.pyplot as plt
import numpy as np

import umap
from MulticoreTSNE import MulticoreTSNE as TSNE
from sklearn.decomposition import PCA
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

#X_fashion = np.load('data/X_fashion.npy')
#y_fashion = np.load('data/y_fashion.npy')
#X_fashion_bin = X_fashion[np.isin(y_fashion, [0, 9])]
#y_fashion_bin = y_fashion[np.isin(y_fashion, [0, 9])]

X_cifar10 = np.load('data/X_cifar10_densenet.npy')
y_cifar10 = np.load('data/y_cifar10.npy')
X_cifar10_bin = X_cifar10[np.isin(y_cifar10, [0, 1])]
y_cifar10_bin = y_cifar10[np.isin(y_cifar10, [0, 1])]

#X_dogsandcats = np.load('data/X_dogsandcats.npy')
#y_dogsandcats = np.load('data/y_dogsandcats.npy')

print("Loading datasets complete")

for label, X, y, p_tsne in zip(#['mnist-bin', 'mnist-full'],
                            ['cifar10-bin', 'cifar10-full'],
                            #['fashion_mnist-bin', 'fashion_mnist-full'],
                            #[X_mnist_bin, X_mnist],
                            #[y_mnist_bin, y_mnist],
                            [X_cifar10_bin, X_cifar10],
                            [y_cifar10_bin, y_cifar10],
                            #[X_fashion_bin, X_fashion],
                            #[y_fashion_bin, y_fashion],
                            #[PCA(n_components=2), PCA(n_components=2), PCA(n_components=2), PCA(n_components=2)]):
                            #[umap.UMAP(n_components=2, random_state=420, n_neighbors=10, min_dist=0.001),
                            #umap.UMAP(n_components=2, random_state=420, n_neighbors=10, min_dist=0.001),
                            #umap.UMAP(n_components=2, random_state=420, n_neighbors=5, min_dist=0.3),
                            #umap.UMAP(n_components=2, random_state=420, n_neighbors=5, min_dist=0.3)]):
                            [TSNE(n_components=2, random_state=420, perplexity=25.0, n_iter=3000, n_iter_without_progress=300, n_jobs=4),
                            TSNE(n_components=2, random_state=420, perplexity=25.0, n_iter=3000, n_iter_without_progress=300, n_jobs=4)]):
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=9000, test_size=3000, random_state=420, stratify=y)

    p = p_tsne
    train_size = 8000

    # A subset is taken to randomize the training a little
    X_train_p, _, y_train_p, _ = train_test_split(X_train, y_train, train_size=train_size, random_state=420, stratify=y_train)
    print("Start seed using " + str(p) + "...")
    t0 = perf_counter()
    X_2d = project(X_train_p, p)
    seed_elapsed_time = perf_counter() - t0
    print("Seeding done in " + str(seed_elapsed_time) + " seconds")

    # Create example plot using the chosen projection function
    X_new = project(X_test, p)
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    fig.tight_layout()
    for x, c in enumerate(np.unique(y_test)):
        ax.axis('off')
        ax.scatter(X_new[y_test==c,0],  X_new[y_test==c,1],  c=cmap(x), s=15, label=c, alpha=0.7)
    #plt.show()
    fig.savefig('Scatterplot_%s_%s.png' % (label, p.__class__.__name__))

    # Data augmentation
    # Todo: "Noise after with sigma = 0.01", ask if this means I offset somewhere in this range or exactly by 0.01 or -0.01?
    std = 0.01
    r_x_list = np.random.uniform(-std, std, len(X_2d))
    r_y_list = np.random.uniform(-std, std, len(X_2d))
    X_2d[:,0] += r_x_list
    X_2d[:,1] += r_y_list

    # Train NNP and save it to a file
    print("Start training...")
    t0 = perf_counter()
    model, hist = nnproj.train_model(X_train_p, X_2d)
    train_elapsed_time = perf_counter() - t0
    epochs = len(hist.history['loss'])
    print("Training complete in " + str(epochs) + " epochs in " + str(train_elapsed_time) + " seconds\n")

    model.save("NNP_model_" + label)
