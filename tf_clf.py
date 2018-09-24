import os

import numpy as np
import tensorflow as tf

from tqdm import tqdm

from lr_utils import load_dataset, iterate_minibatches

X_train, Y_train, X_test, Y_test, X_val, Y_val, classes = load_dataset()
# X_train, Y_train = np.random.random(size=(1000, 256, 256, 3)).astype(np.float32), np.random.randint(2, size=(1000, 1)).astype(np.float32)
# X_test, Y_test = np.random.random(size=(200, 256, 256, 3)).astype(np.float32), np.random.randint(2, size=(200, 1)).astype(np.float32)
# X_val, Y_val = np.random.random(size=(100, 256, 256, 3)).astype(np.float32), np.random.randint(2, size=(100, 1)).astype(np.float32)

print ("number of training examples = " + str(X_train.shape[0]))
print ("number of test examples = " + str(X_test.shape[0]))
print ("X_train shape: " + str(X_train.shape))
print ("Y_train shape: " + str(Y_train.shape))
print ("X_test shape: " + str(X_test.shape))
print ("Y_test shape: " + str(Y_test.shape))

num_examples = X_train.shape[0]
input_shape = (None, ) + tuple(X_train.shape[1: ])

tf.reset_default_graph()

datatype = tf.float32
initializer = tf.random_uniform_initializer()

image_data = tf.placeholder(dtype=datatype, shape=input_shape)
targets = tf.placeholder(dtype=tf.uint8, shape=(None, 1))
keep_prob = tf.placeholder(dtype=datatype)

zero_padding_1 = tf.pad(image_data, [[0, 0], [3, 3], [3, 3], [0, 0]])

kernel = tf.get_variable('kernel', shape=[7, 7, 3, 32], initializer=initializer, dtype=datatype)
conv_1 = tf.nn.conv2d(zero_padding_1, filter=kernel, strides=[1, 1, 1, 1], padding='SAME')
bn_1 = tf.layers.batch_normalization(conv_1)
relu_1 = tf.nn.relu(conv_1)
dropout_1 = tf.nn.dropout(relu_1, keep_prob=keep_prob)
max_pool_1 = tf.nn.max_pool(relu_1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

dim = np.prod(max_pool_1.get_shape().as_list()[1: ])
flatten_1 = tf.reshape(max_pool_1, shape=[-1, dim])
# flatten_1 = tf.layers.flatten(max_pool_1)

weights = tf.get_variable('weights', shape=[dim, 1], initializer=initializer, dtype=datatype)
bias = tf.get_variable('bias', shape=[1], initializer=initializer, dtype=datatype)
dense_1 = tf.add(tf.matmul(flatten_1, weights), bias)
# output = tf.layers.dense(flatten_1, 1, activation='sigmoid')

output = tf.nn.sigmoid(dense_1)

loss = tf.losses.sigmoid_cross_entropy(targets, logits=output)
optimizer = tf.train.AdamOptimizer()
train_step = optimizer.minimize(loss)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

epochs = 10
batch_size = 32

n_steps = num_examples // batch_size
if not num_examples % batch_size == 0:
    n_steps += 1

for epoch in range(epochs):
    # Training
    total_train_loss, n_iter = 0.0, 0
    for image_batch, label_batch in tqdm(iterate_minibatches(X_train, Y_train,
        batchsize=batch_size, shuffle=True), total=n_steps, 
        desc='Epoch {}/{}'.format(epoch, epochs)):
        _, train_loss_iter = sess.run([train_step, loss],
                                feed_dict={image_data: image_batch,
                                            targets: label_batch,
                                            keep_prob: 0.5})
        total_train_loss += train_loss_iter
        n_iter += 1

    avg_train_loss = total_train_loss / n_iter

    # Validation
    total_val_loss, n_iter = 0.0, 0
    for image_batch, label_batch in iterate_minibatches(X_val, Y_val,
        batchsize=batch_size, shuffle=True):
        val_loss_iter = sess.run([loss], feed_dict={image_data: image_batch,
                                                targets: label_batch,
                                                keep_prob: 1.0})
        total_val_loss += val_loss_iter[0]
        n_iter += 1

    avg_val_loss = total_val_loss / n_iter
    print('Epoch[{}/{}]   Training Loss: {:.4f}   Validation Loss: {:.4f}'
        .format(epoch, epochs, avg_train_loss, avg_val_loss))

# Testing
total_test_loss, n_iter = 0.0, 0
for image_batch, label_batch in iterate_minibatches(X_test, Y_test,
    batchsize=batch_size, shuffle=True):
    test_loss_iter = sess.run([loss],feed_dict={image_data: image_batch,
                                                targets: label_batch,
                                                keep_prob: 1.0})
    total_test_loss += test_loss_iter[0]
    n_iter += 1

avg_test_loss = total_test_loss / n_iter
print('Test Loss: {:.4f}'.format(avg_test_loss))
