import os
from keras.preprocessing import image
from keras.utils import layer_utils
from keras.utils.data_utils import get_file
from keras.applications.imagenet_utils import preprocess_input
import numpy as np
import h5py

def load_dataset():
    train_dataset = h5py.File('datasets/training_data.h5', 'r')
    X_train = np.array(train_dataset["X_train"][:]) # your train set features
    y_train = np.array(train_dataset["y_train"][:]) # your train set labels

    train_dataset = h5py.File('datasets/validation_data.h5', 'r')
    X_val = np.array(train_dataset["X_val"][:]) # your validation set features
    y_val = np.array(train_dataset["y_val"][:]) # your validation set labels

    train_dataset = h5py.File('datasets/testing_data.h5', 'r')
    X_test = np.array(train_dataset["X_test"][:]) # your test set features
    y_test = np.array(train_dataset["y_test"][:]) # your test set labels

    return X_train, X_val, X_test, y_train, y_val, y_test

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