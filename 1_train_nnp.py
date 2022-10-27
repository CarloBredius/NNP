#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
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
print("Loading datasets complete")

# Determine amount of training samples
number_of_samples = int(sys.argv[1])
print("Amount of samples: " + str(number_of_samples))

# Function to use
p = TSNE(n_components=2, random_state=420, perplexity=25.0, n_iter=3000, n_iter_without_progress=300, n_jobs=4)
y = 10
X_train, X_test, y_train, y_test = train_test_split(X_mnist, y_mnist, train_size=10000, test_size=3000, random_state=420, stratify=y_mnist)

# A subset is taken to randomize the training a little
X_train_p, _, y_train_p, _ = train_test_split(X_train, y_train, train_size=number_of_samples, random_state=420, stratify=y_train)
print(f"Start seed using {str(p)}...")
t0 = perf_counter()
X_2d = project(X_train_p, p)
seed_elapsed_time = perf_counter() - t0
print(f"Seeding done in {str(seed_elapsed_time)} seconds")

# Data augmentation
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
print(f"Training complete in {str(epochs)} epochs in {str(train_elapsed_time)} seconds\n")

label = "mnist-full"
model.save(f"NNP_model_{label}_" + str(number_of_samples))

# Predict using model
X_2d_pred = model.predict(X_test)

# Create scatter-plot using NNP
fig2, ax2 = plt.subplots(1, 1, figsize=(10, 10))
fig2.tight_layout()
for x, c in enumerate(np.unique(y_train_p)):
	print(c)
	print(x)

	ax2.axis('off')
	ax2.scatter(X_2d_pred[y_test==c,0],  X_2d_pred[y_test==c,1],  c=cmap(x), s=15, label=c, alpha=0.7)
fig2.savefig(f"projection_{label}_{str(number_of_samples)}.png")
