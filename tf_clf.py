import os

import numpy as np
import tensorflow as tf

from lr_utils import load_dataset

X_train, Y_train, X_test, Y_test, X_val, Y_val, classes = load_dataset()

print ("number of training examples = " + str(X_train.shape[0]))
print ("number of test examples = " + str(X_test.shape[0]))
print ("X_train shape: " + str(X_train.shape))
print ("Y_train shape: " + str(Y_train.shape))
print ("X_test shape: " + str(X_test.shape))
print ("Y_test shape: " + str(Y_test.shape))

input_shape = (None, ) + tuple(X_train.shape[1: ])

tf.reset_default_graph()

datatype = tf.float32
initializer = tf.random_uniform_initializer()

image_data = tf.placeholder(dtype=datatype, shape=input_shape)
targets = tf.placeholder(dtype=tf.uint8, shape=(None, 1))

padding = tf.constant([[1, 3], [2, 3]], dtype=datatype)
zero_padding_1 = tf.pad(image_data, padding)

kernel = tf.get_variable('kernel', shape=[7, 7, 3, 32], initializer=initializer, dtype=datatype)
conv_1 = tf.nn.conv2d(zero_padding_1, filter=kernel, strides=[1, 1], padding='SAME')
# bn_1 = tf.nn.batch_normalization(conv_1)
relu_1 = tf.nn.relu(conv_1)

max_pool_1 = tf.nn.max_pool(relu_1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

flatten_1 = tf.reshape(max_pool_1, shape=[None, -1])

# weights = tf.get_variable('weights', shape=[flatten_1.shape[1], 1], initializer=initializer, dtype=datatype)
# bias = tf.get_variable('bias', shape=[1], initializer=initializer, dtype=datatype)
# dense_1 = tf.add(tf.matmul(flatten_1, weights), bias)
output = tf.layers.dense(flatten_1, 1, activation='sigmoid')

loss = tf.losses.sigmoid_cross_entropy(targets, logits=output)
optimizer = tf.train.AdamOptimizer()
train_step = optimizer.minimize(loss)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

for i in range(10):

    random_image, random_label = tf.train.slice_input_producer([X_train, Y_train], shuffle=True)
    image_batch, label_batch = tf.train.batch([random_image, random_label], batch_size=50)
    
    _, loss_iter = sess.run([train_step, loss],
                            feed_dict={image_data: image_batch,
                                        targets: label_batch})
    print(i, loss_iter)