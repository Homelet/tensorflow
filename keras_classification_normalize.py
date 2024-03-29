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

# x = (x - u) / std

from sklearn.preprocessing import StandardScaler


scalar = StandardScaler()

# x_train [None, 28, 28] -> [None, 784]
x_train_scaled = scalar.fit_transform(x_train.astype(np.float32).reshape(-1, 1)).reshape(-1, 28, 28)
x_valid_scaled = scalar.transform(x_valid.astype(np.float32).reshape(-1, 1)).reshape(-1, 28, 28)
x_test_scaled = scalar.transform(x_test.astype(np.float32).reshape(-1, 1)).reshape(-1, 28, 28)

class_name = ["T-shirt", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]

# show_multi_image(3, 5, x_train, y_train, class_name)

module = keras.models.Sequential()

module.add(keras.layers.Flatten(input_shape=[28, 28]))

module.add(keras.layers.Dense(300, activation="relu"))

module.add(keras.layers.Dense(100, activation="relu"))

module.add(keras.layers.Dense(10, activation="softmax"))

# reason for sparse: y->index, y->one_hot->[] s
module.compile(loss="sparse_categorical_crossentropy", optimizer="sgd", metrics=["accuracy"])

history = module.fit(x_train_scaled, y_train, epochs=10, validation_data=(x_valid_scaled, y_valid))


def plot_learning_curve(history):
	pd.DataFrame(history.history).plot(figsize=(8, 5))
	plt.grid(True)
	plt.gca().set_ylim(0, 1)
	plt.show()


plot_learning_curve(history)
