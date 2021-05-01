#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
from time import perf_counter

import joblib
import matplotlib.pyplot as plt
import numpy as np
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

#X_fashion = np.load('data/X_fashion.npy')
#y_fashion = np.load('data/y_fashion.npy')
#X_fashion_bin = X_fashion[np.isin(y_fashion, [0, 9])]
#y_fashion_bin = y_fashion[np.isin(y_fashion, [0, 9])]

#X_dogsandcats = np.load('data/X_dogsandcats.npy')
#y_dogsandcats = np.load('data/y_dogsandcats.npy')

print("Loading datasets complete")

for label, X, y, p_tsne in zip(['mnist-bin', 'mnist-full'],
                            [X_mnist_bin, X_mnist],
                            [y_mnist_bin, y_mnist],
                            [TSNE(n_components=2, random_state=420, perplexity=25.0, n_iter=3000, n_iter_without_progress=300, n_jobs=4),
                            TSNE(n_components=2, random_state=420, perplexity=25.0, n_iter=3000, n_iter_without_progress=300, n_jobs=4)]):
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=10000, test_size=3000, random_state=420, stratify=y)

    p = p_tsne
    train_size = 9000

    #print("Project using", p)
    #t0 = perf_counter()    
    #X_new = project(X_test, p)
    #proj_elapsed_time = perf_counter() - t0

    # Create scatter-plot using t-SNE
    #fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    #fig.tight_layout()
    #for x, c in enumerate(np.unique(y_test)):
    #    ax.axis('off')
    #    ax.scatter(X_new[y_test==c,0],  X_new[y_test==c,1],  c=cmap(x), s=15, label=c, alpha=0.7)
    #fig.savefig('Scatterplot_%s_%s.png' % (label, p.__class__.__name__))
    #plt.show()
    #print("\n" + label, "- projected with", p.__class__.__name__ + ", %d samples" % train_size)

    # A subset is taken to randomize the training a little?
    X_train_p, _, y_train_p, _ = train_test_split(X_train, y_train, train_size=train_size, random_state=420, stratify=y_train)

    t0 = perf_counter()
    X_2d = project(X_train_p, p)
    seed_elapsed_time = perf_counter() - t0

    # Train NNP and save it to a file
    print("Start training...")
    t0 = perf_counter()
    model, hist = nnproj.train_model(X_train_p, X_2d)
    train_elapsed_time = perf_counter() - t0
    epochs = len(hist.history['loss'])
    print("Training complete in " + str(epochs) + " epochs\n")

    model.save("NNP_model_" + label)

    ## Use NNP to project
    #t0 = perf_counter()
    #X_2d_pred = model.predict(X_test)
    #infer_elapsed_time = perf_counter() - t0

    ## Usable variables: proj_elapsed_time, seed_elapsed_time, train_elapsed_time, infer_elapsed_time, X_2d_pred, X_2d, X_new, model, hist, epochs
    #print('%.s: Project: %.2f\nNNP with t-SNE: seed: %.2f, train: %.2f, infer: %.2f' % (p, proj_elapsed_time, seed_elapsed_time, train_elapsed_time, infer_elapsed_time))

    ## Create scatter-plot using NNP
    #fig2, ax2 = plt.subplots(1, 1, figsize=(10, 10))
    #fig2.tight_layout()
    #for x, c in enumerate(np.unique(y_train_p)):
    #    ax2.axis('off')
    #    ax2.scatter(X_2d_pred[y_test==c,0],  X_2d_pred[y_test==c,1],  c=cmap(x), s=15, label=c, alpha=0.7)
    #fig2.savefig('Scatterplot_NNP_%s_%s.png' % (label, p.__class__.__name__))
