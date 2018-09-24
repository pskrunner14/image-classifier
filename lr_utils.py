import os
from keras.preprocessing import image
from keras.utils import layer_utils
from keras.utils.data_utils import get_file
from keras.applications.imagenet_utils import preprocess_input
import numpy as np
import h5py

def load_dataset():
    train_dataset = h5py.File('datasets/training_data.h5', "r")
    x_train = np.array(train_dataset["x_train"]) # your train set features
    y_train = np.array(train_dataset["y_train"]) # your train set labels
    labels = np.array(train_dataset["labels"])

    test_dataset = h5py.File('datasets/testing_data.h5', "r")
    x_test = np.array(test_dataset["x_test"]) # your test set features
    y_test = np.array(test_dataset["y_test"]) # your test set labels

    x_val = x_train[-1000: ]
    y_val = y_train[-1000: ]

    x_train = x_train[: -1000]
    y_train = y_train[: -1000]

    return x_train, y_train, x_test, y_test, x_val, y_val, labels

def iterate_minibatches(inputs, targets, batchsize, shuffle=False):
    assert len(inputs) == len(targets)
    if shuffle:
        indices = np.random.permutation(len(inputs))
    for start_idx in range(0, len(inputs) - batchsize + 1, batchsize):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batchsize]
        else:
            excerpt = slice(start_idx, start_idx + batchsize)
        yield inputs[excerpt], targets[excerpt]