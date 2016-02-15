# -*- coding: UTF-8 -*-
"""
Script for assignment 4.
"""

from __future__ import print_function

import os

import numpy as np
import tensorflow as tf
import cPickle as pickle

PROJECT_ROOT = '/Users/kensk8er/PycharmProjects/udacity'
PICKLE_FILE = os.path.join(PROJECT_ROOT, 'notMNIST.pickle')

# assignment parameters
IMAGE_SIZE = 28
NUM_LABELS = 10
NUM_CHANNELS = 1  # greyscale

# CNN parameters
BATCH_SIZE = 16
PATCH_SIZE = 5
DEPTH = 16  # the number of the convolution-layer's feature maps?
NUM_HIDDEN = 64  # the number of neurons for fully-connected layer?
LEARNING_RATE = 0.05
NUM_STEPS = 1001

__author__ = 'kensk8er'


def load_data(file_path):
    with open(file_path, 'rb') as pickle_file:
        save = pickle.load(pickle_file)
        train_dataset = save['train_dataset']
        train_labels = save['train_labels']
        valid_dataset = save['valid_dataset']
        valid_labels = save['valid_labels']
        test_dataset = save['test_dataset']
        test_labels = save['test_labels']
        del save  # hint to help gc free up memory
        print('Training set', train_dataset.shape, train_labels.shape)
        print('Validation set', valid_dataset.shape, valid_labels.shape)
        print('Test set', test_dataset.shape, test_labels.shape)

        return train_dataset, train_labels, valid_dataset, valid_labels, test_dataset, test_labels


def reformat(dataset, labels):
    dataset = dataset.reshape((-1, IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS)).astype(np.float32)
    labels = (np.arange(NUM_LABELS) == labels[:, None]).astype(np.float32)
    return dataset, labels


def accuracy(predictions, labels):
    return 100. * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1)) / predictions.shape[0]


if __name__ == '__main__':
    # load data
    train_dataset, train_labels, valid_dataset, valid_labels, test_dataset, test_labels = load_data(PICKLE_FILE)

    # reformat data
    train_dataset, train_labels = reformat(train_dataset, train_labels)
    valid_dataset, valid_labels = reformat(valid_dataset, valid_labels)
    test_dataset, test_labels = reformat(test_dataset, test_labels)
    print('Training set', train_dataset.shape, train_labels.shape)
    print('Validation set', valid_dataset.shape, valid_labels.shape)
    print('Test set', test_dataset.shape, test_labels.shape)

    # Create CNN
    graph = tf.Graph()
    with graph.as_default():
        # Input data.
        tf_train_dataset = tf.placeholder(tf.float32, shape=(BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS))
        tf_train_labels = tf.placeholder(tf.float32, shape=(BATCH_SIZE, NUM_LABELS))
        tf_valid_dataset = tf.constant(valid_dataset)
        tf_test_dataset = tf.constant(test_dataset)

        # Variables.
        layer1_weights = tf.Variable(tf.truncated_normal([PATCH_SIZE, PATCH_SIZE, NUM_CHANNELS, DEPTH], stddev=0.1))
        layer1_biases = tf.Variable(tf.zeros([DEPTH]))

        layer2_weights = tf.Variable(tf.truncated_normal([PATCH_SIZE, PATCH_SIZE, DEPTH, DEPTH], stddev=0.1))
        layer2_biases = tf.Variable(tf.constant(1.0, shape=[DEPTH]))  # why initialized as 1.0??

        layer3_weights = tf.Variable(tf.truncated_normal(
            [IMAGE_SIZE // 4 * IMAGE_SIZE // 4 * DEPTH, NUM_HIDDEN], stddev=0.1))
        layer3_biases = tf.Variable(tf.constant(1.0, shape=[NUM_HIDDEN]))  # why initialized as 1.0??

        layer4_weights = tf.Variable(tf.truncated_normal([NUM_HIDDEN, NUM_LABELS], stddev=0.1))
        layer4_biases = tf.Variable(tf.constant(1.0, shape=[NUM_LABELS]))  # why initialized as 1.0??

        # Model.
        def model(data):
            # 1st (convolution)
            convolution = tf.nn.conv2d(data, layer1_weights, [1, 2, 2, 1], padding='SAME')
            hidden = tf.nn.relu(convolution + layer1_biases)

            # 2nd (convolution)
            convolution = tf.nn.conv2d(hidden, layer2_weights, [1, 2, 2, 1], padding='SAME')
            hidden = tf.nn.relu(convolution + layer2_biases)

            # 3rd (fully-connected hidden layer)
            shape = hidden.get_shape().as_list()
            reshape = tf.reshape(hidden, [shape[0], shape[1] * shape[2] * shape[3]])  # convert to 1-D?
            hidden = tf.nn.relu(tf.matmul(reshape, layer3_weights) + layer3_biases)

            # 4th (final softmax layer)
            return tf.matmul(hidden, layer4_weights) + layer4_biases


        # Training computation.
        logits = model(tf_train_dataset)
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits, tf_train_labels))

        # Optimizer.
        optimizer = tf.train.GradientDescentOptimizer(LEARNING_RATE).minimize(loss)

        # Predictions for the training, validation, and test data.
        train_prediction = tf.nn.softmax(logits)
        valid_prediction = tf.nn.softmax(model(tf_valid_dataset))
        test_prediction = tf.nn.softmax(model(tf_test_dataset))

    # Run the CNN defined
    with tf.Session(graph=graph) as session:
        tf.initialize_all_variables().run()
        print('Initialized')

        for step in range(NUM_STEPS):
            offset = (step * BATCH_SIZE) % (train_labels.shape[0] - BATCH_SIZE)
            batch_data = train_dataset[offset: (offset + BATCH_SIZE), :, :, :]
            batch_labels = train_labels[offset: (offset + BATCH_SIZE), :]
            feed_dict = {tf_train_dataset : batch_data, tf_train_labels : batch_labels}

            _, l, predictions = session.run([optimizer, loss, train_prediction], feed_dict=feed_dict)

            if step % 50 == 0:
                print('Minibatch loss at step %d: %f' % (step, l))
                print('Minibatch accuracy: %.1f%%' % accuracy(predictions, batch_labels))
                print('Validation accuracy: %.1f%%' % accuracy(valid_prediction.eval(), valid_labels))

        print('Test accuracy: %.1f%%' % accuracy(test_prediction.eval(), test_labels))
