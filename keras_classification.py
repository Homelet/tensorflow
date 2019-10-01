import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import sklearn
import pandas as pd
import os
import time
import sys
import tensorflow as tf
from tensorflow.python import keras


print(tf.__version__)
print(sys.version_info)

for module in mpl, np, pd, sklearn, tf, keras:
	print(module.__name__, module.__version__)

fashion_mnist = keras.datasets.fashion_mnist

(x_train_all, y_train_all), (x_test, y_test) = fashion_mnist.load_data()

x_valid, x_train, = x_train_all[:5000], x_train_all[5000:]
y_valid, y_train, = y_train_all[:5000], y_train_all[5000:]

print(x_valid.shape, y_valid.shape)
print(x_train.shape, y_train.shape)
print(x_test.shape, y_test.shape)


def show_single_image(image):
	plt.imshow(image, cmap="binary")
	plt.show()


def show_multi_image(n_row, n_col, x_data, y_data, class_name):
	assert len(x_data) == len(y_data)
	assert n_row * n_col < len(x_data)
	plt.figure(figsize=(n_col * 1.4, n_row * 1.6))
	for row in range(n_row):
		for col in range(n_col):
			index = n_col * row + col
			plt.subplot(n_row, n_col, index + 1)
			plt.imshow(x_data[index], cmap="binary", interpolation="nearest")
			plt.axis("off")
			plt.title(class_name[y_data[index]])
	
	plt.show()


class_name = ["T-shirt", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]

# show_multi_image(3, 5, x_train, y_train, class_name)

module = keras.models.Sequential()

module.add(keras.layers.Flatten(input_shape=[28, 28]))

module.add(keras.layers.Dense(300, activation="relu"))

module.add(keras.layers.Dense(100, activation="relu"))

module.add(keras.layers.Dense(10, activation="softmax"))

# reason for sparse: y->index, y->one_hot->[] s
module.compile(loss="sparse_categorical_crossentropy", optimizer="sgd", metrics=["accuracy"])

module.summary()

history = module.fit(x_train, y_train, epochs=10, validation_data=(x_valid, y_valid))


def plot_learning_curve(history):
	pd.DataFrame(history.history).plot(figsize=(8, 5))
	plt.grid(True)
	plt.gca().set_ylim(0, 1)
	plt.show()


plot_learning_curve(history )
