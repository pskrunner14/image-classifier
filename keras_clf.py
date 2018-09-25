import os
import numpy as np
import keras

from lr_utils import load_dataset

import keras.backend as K
K.set_image_data_format('channels_last')

import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow

X_train, X_val, X_test, Y_train, Y_val, Y_test = load_dataset()

print ("number of training examples = " + str(X_train.shape[0]))
print ("number of test examples = " + str(X_test.shape[0]))
print ("X_train shape: " + str(X_train.shape))
print ("Y_train shape: " + str(Y_train.shape))
print ("X_test shape: " + str(X_test.shape))
print ("Y_test shape: " + str(Y_test.shape))

def create_model(input_shape):
    X_input = keras.layers.Input(input_shape)
    
    # Zero-Padding: pads the border of X_input with zeroes
    X = keras.layers.ZeroPadding2D((3, 3))(X_input)
    
    # CONV -> BN -> RELU Block applied to X
    X = keras.layers.Conv2D(32, (7, 7), strides = (1, 1), name = 'conv0')(X)
    X = keras.layers.BatchNormalization(axis = 3, name = 'bn0')(X)
    X = keras.layers.Activation('relu')(X)
    X = keras.layers.Dropout(0.5)(X)
    
    # MAXPOOL
    X = keras.layers.MaxPooling2D((2, 2), name='max_pool')(X)
    
    # FLATTEN X (means convert it to a vector) + FULLYCONNECTED
    X = keras.layers.Flatten()(X)
    X = keras.layers.Dense(1, activation='sigmoid', name='fc')(X)

    # Create model. This creates your Keras model instance, you'll use this instance to train/test the model.
    model = keras.models.Model(inputs = X_input, outputs = X)
    return model

# Build model
model = create_model(X_train.shape[1:])
model.compile(optimizer = "adam", loss = "binary_crossentropy", metrics = ["accuracy"])
model.summary()

# Train model
model.fit(x=X_train, y=Y_train, validation_data=(X_val, Y_val), 
            shuffle=True, epochs=25, batch_size=32, verbose=1)

# Save model
model.save('models/keras_model.hd5')
preds = model.evaluate(x=X_test, y=Y_test, verbose=1)

print ("Test Loss: {}".format(preds[0]))
print ("Test Accuracy: {}".format(preds[1]))

keras.utils.plot_model(model, to_file='models/keras_model.png')

def make_prediction(input_image, true_labal):
    if true_labal == 1:
        print("This picture contains a cat")
    else:
        print("This picture does'nt contain a cat")

    preds = model.predict(input_image)[0][0]
    class_pred = 'cat' if int(preds) == 1 else 'non-cat'
    print('y = {}, algorithm predicts a \"{}\" picture.'.format(preds, class_pred))