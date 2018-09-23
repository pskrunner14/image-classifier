import os
import numpy as np
from keras import layers
from keras.layers import Input, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D
from keras.layers import AveragePooling2D, MaxPooling2D, Dropout, GlobalMaxPooling2D, GlobalAveragePooling2D
from keras.models import Model
from keras.preprocessing import image
from keras.utils import layer_utils
from keras.utils.data_utils import get_file
from keras.applications.imagenet_utils import preprocess_input
import pydot
from keras.utils.vis_utils import model_to_dot
from keras.utils import plot_model
from lr_utils import load_dataset

import keras.backend as K
K.set_image_data_format('channels_last')

import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow

X_train, Y_train, X_test, Y_test, X_val, Y_val, classes = load_dataset()

print ("number of training examples = " + str(X_train.shape[0]))
print ("number of test examples = " + str(X_test.shape[0]))
print ("X_train shape: " + str(X_train.shape))
print ("Y_train shape: " + str(Y_train.shape))
print ("X_test shape: " + str(X_test.shape))
print ("Y_test shape: " + str(Y_test.shape))

def create_model(input_shape):

    # Define the input placeholder as a tensor with shape input_shape. Think of this as your input image!
    X_input = Input(input_shape)

    # Zero-Padding: pads the border of X_input with zeroes
    X = ZeroPadding2D((3, 3))(X_input)

    # CONV -> BN -> RELU Block applied to X
    X = Conv2D(32, (7, 7), strides = (1, 1), name = 'conv0')(X)
    X = BatchNormalization(axis = 3, name = 'bn0')(X)
    X = Activation('relu')(X)

    # MAXPOOL
    X = MaxPooling2D((2, 2), name='max_pool')(X)

    # FLATTEN X (means convert it to a vector) + FULLYCONNECTED
    X = Flatten()(X)
    X = Dense(1, activation='sigmoid', name='fc')(X)

    # Create model. This creates your Keras model instance, you'll use this instance to train/test the model.
    model = Model(inputs = X_input, outputs = X, name='Cat Detector')

    return model

# Build model
model = create_model(X_train.shape[1:])
model.compile(optimizer = "adam", loss = "binary_crossentropy", metrics = ["accuracy"])
model.summary()

# Train model
model.fit(x = X_train, y = Y_train, epochs = 10, batch_size = 50, verbose = 1)

# Save model
model.save('models/keras_model.hd5')

preds = model.evaluate(x = X_test, y = Y_test, verbose  = 1)

print ("Test Loss: {}".format(preds[0]))
print ("Test Accuracy: {}".format(preds[1]))

for i in range(len(X_val)):
    print("Image #{}".format(i))
    if Y_val[i] == 1:
        print("This picture contains a cat")
    else:
        print("This picture does'nt contain a cat")

    prediction = model.predict(X_val[i])[0][0]
    print('y = {}, algorithm predicts a \"{}\" picture.'.format(prediction, classes[int(prediction)].decode('utf-8')))
    print()
    index += 1

plot_model(model, to_file='models/keras_model.png')