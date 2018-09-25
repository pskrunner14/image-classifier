import os
import time

import click
import numpy as np
import tensorflow as tf

from tqdm import tqdm

from utils import load_dataset, iterate_minibatches

"""
Training model [optional args]
"""
@click.command()
@click.option('-n', '--num-epochs', default=15, help='Number of epochs for training',)
@click.option('-bs', '--batch-size', default=32, help='Batch size for training on minibatches',)
@click.option('-lr', '--learning-rate', default=0.001, help='Learning rate to use when training model',)
@click.option('-tb', '--tensorboard-vis', is_flag=True, help='Flag for TensorBoard visualization',)
def train(num_epochs, batch_size, learning_rate, tensorboard_vis):

    X_train, X_val, X_test, Y_train, Y_val, Y_test = load_dataset()
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
    timestamp = time.strftime('%Y_%m_%d_%H_%M_%S', time.localtime())

    tf.reset_default_graph()

    image_data = tf.placeholder(dtype=tf.float32, shape=input_shape, name='image_data')
    targets = tf.placeholder(dtype=tf.float32, shape=(None, 1), name='targets')
    keep_prob = tf.placeholder(dtype=tf.float32, name='keep_prob')

    with tf.variable_scope('zero_pad') as scope:
        zero_pad = tf.pad(image_data, [[0, 0], [3, 3], [3, 3], [0, 0]], name=scope.name)

    with tf.variable_scope('conv1') as scope:
        kernel = tf.get_variable('kernel', shape=[7, 7, 3, 32], 
                                initializer=tf.random_uniform_initializer(), 
                                dtype=tf.float32)
        conv = tf.nn.conv2d(zero_pad, filter=kernel, 
                            strides=[1, 1, 1, 1], 
                            padding='SAME')
        bn = tf.layers.batch_normalization(conv)
        relu = tf.nn.relu(bn)
        dropout = tf.nn.dropout(relu, keep_prob=keep_prob)
        conv1 = tf.nn.max_pool(dropout, ksize=[1, 2, 2, 1], 
                                strides=[1, 2, 2, 1], padding='SAME', 
                                name=scope.name)

    with tf.variable_scope('logits') as scope:
        dim = np.prod(conv1.get_shape().as_list()[1: ])
        flatten = tf.reshape(conv1, shape=[-1, dim])
        weights = tf.get_variable('weights', shape=[dim, 1], 
                    initializer=tf.random_uniform_initializer(), 
                    dtype=tf.float32)
        bias = tf.get_variable('bias', shape=[1], 
                initializer=tf.constant_initializer(0.0), 
                dtype=tf.float32)
        dense = tf.add(tf.matmul(flatten, weights), bias)
        logits = tf.nn.sigmoid(dense, name=scope.name)

    loss = tf.losses.sigmoid_cross_entropy(targets, logits=logits)
    accuracy = tf.reduce_mean(tf.cast(tf.equal(logits, targets), dtype=tf.float32))

    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    train_step = optimizer.minimize(loss)

    if tensorboard_vis:
        tf.summary.scalar('loss', loss)
        tf.summary.scalar('accuracy', accuracy)
        summaries = tf.summary.merge_all()

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    saver = tf.train.Saver(tf.trainable_variables())

    n_steps = num_examples // batch_size
    if not num_examples % batch_size == 0:
        n_steps += 1

    if tensorboard_vis:
        train_writer = tf.summary.FileWriter('logs/train', sess.graph)
        val_writer = tf.summary.FileWriter('logs/val', sess.graph)

    for epoch in range(num_epochs):
        # Training
        train_losses, train_accuracies, n_iter = [], [], 0
        for image_batch, label_batch in tqdm(iterate_minibatches(X_train, Y_train,
            batchsize=batch_size, shuffle=True), total=n_steps, 
            desc='Epoch {}/{}'.format(epoch, num_epochs)):
            if tensorboard_vis:
                _, train_loss, train_acc, summary = sess.run([train_step, loss, accuracy, summaries],
                                                feed_dict={image_data: image_batch,
                                                            targets: label_batch,
                                                            keep_prob: 0.5})
            else:
                _, train_loss, train_acc = sess.run([train_step, loss, accuracy],
                                                feed_dict={image_data: image_batch,
                                                            targets: label_batch,
                                                            keep_prob: 0.5})
            if tensorboard_vis and n_iter == 0:
                train_writer.add_summary(summary, n_iter)
                train_writer.flush()

            train_losses.append(train_loss)
            train_accuracies.append(train_acc)
            n_iter += 1

        avg_train_loss = np.mean(train_losses)
        avg_train_acc = np.mean(train_accuracies)

        # Validation
        val_losses, val_accuracies, n_iter = [], [], 0
        for image_batch, label_batch in iterate_minibatches(X_val, Y_val,
            batchsize=batch_size, shuffle=True):
            if tensorboard_vis:
                val_loss, val_acc, summary = sess.run([loss, accuracy, summaries], 
                                            feed_dict={image_data: image_batch,
                                                        targets: label_batch,
                                                        keep_prob: 1.0})
            else:
                val_loss, val_acc = sess.run([loss, accuracy], 
                                            feed_dict={image_data: image_batch,
                                                        targets: label_batch,
                                                        keep_prob: 1.0})
            if tensorboard_vis and n_iter == 0:
                val_writer.add_summary(summaries, n_iter)
                val_writer.flush()
            val_losses.append(val_loss)
            val_accuracies.append(val_acc)
            n_iter += 1

        avg_val_loss = np.mean(val_losses)
        avg_val_acc = np.mean(val_accuracies)
        print('Epoch {}/{}: train loss: {:.4f} train acc: {:.4f} val loss: {:.4f} val acc: {:.4f}'
            .format(epoch, num_epochs, avg_train_loss, avg_train_acc, avg_val_loss, avg_val_acc))
        # save model checkpoint
        saver.save(sess, 'models/{}/model.ckpt'
            .format(timestamp), global_step=epoch)

    # Testing
    test_losses, test_accuracies, n_iter = [], [], 0
    for image_batch, label_batch in iterate_minibatches(X_test, Y_test,
        batchsize=batch_size, shuffle=True):
        test_loss, test_acc = sess.run([loss, accuracy],feed_dict={image_data: image_batch,
                                                    targets: label_batch,
                                                    keep_prob: 1.0})
        test_losses.append(test_loss)
        test_accuracies.append(test_acc)
        n_iter += 1

    avg_test_loss = np.mean(test_losses)
    avg_test_acc = np.mean(test_accuracies)
    print('Test Loss: {:.4f} Test Accuracy: {:.4f}'.format(avg_test_loss, avg_test_acc))

    sess.close()

def main():
    try:
        train()
    except KeyboardInterrupt:
        print('EXIT')

if __name__ == '__main__':
    main()