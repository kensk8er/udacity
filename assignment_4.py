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
C_1_PATCH_SIZE = 5
C_1_MAP_NUM = 12  # the number of the feature maps for C-1 layer
S_2_PATCH_SIZE = 2
C_3_PATCH_SIZE = 5
C_3_MAP_NUM = 32  # the number of the feature maps for C-3 layer
S_4_PATCH_SIZE = 2
C_5_NEURON_NUM = 240  # the number of neurons for C-5 layer
F_6_NEURON_NUM = 168  # the number of neurons for F-6 layer
INITIAL_LEARNING_RATE = 0.05
NUM_STEPS = 25001
# NUM_STEPS = 1001
DROPOUT_PROBABILITY = 0.5

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
        c_1_weights = tf.Variable(tf.truncated_normal([C_1_PATCH_SIZE, C_1_PATCH_SIZE, NUM_CHANNELS, C_1_MAP_NUM],
                                                      stddev=0.1))
        c_1_biases = tf.Variable(tf.zeros([C_1_MAP_NUM]))

        c_3_weights = tf.Variable(tf.truncated_normal([C_3_PATCH_SIZE, C_3_PATCH_SIZE, C_1_MAP_NUM, C_3_MAP_NUM],
                                                      stddev=0.1))
        c_3_biases = tf.Variable(tf.constant(1.0, shape=[C_3_MAP_NUM]))  # why initialized as 1.0??

        c_5_feature_size = ((IMAGE_SIZE - C_1_PATCH_SIZE + 1) // 2 - C_3_PATCH_SIZE + 1) // 2
        c_5_weights = tf.Variable(tf.truncated_normal(
            [c_5_feature_size * c_5_feature_size * C_3_MAP_NUM, C_5_NEURON_NUM], stddev=0.1))
        c_5_biases = tf.Variable(tf.constant(1.0, shape=[C_5_NEURON_NUM]))  # why initialized as 1.0??

        f_6_weights = tf.Variable(tf.truncated_normal([C_5_NEURON_NUM, F_6_NEURON_NUM], stddev=0.1))
        f_6_biases = tf.Variable(tf.constant(1.0, shape=[F_6_NEURON_NUM]))

        output_weights = tf.Variable(tf.truncated_normal([F_6_NEURON_NUM, NUM_LABELS], stddev=0.1))
        output_biases = tf.Variable(tf.constant(1.0, shape=[NUM_LABELS]))  # why initialized as 1.0??

        global_step = tf.Variable(0, trainable=False)
        learning_rate = tf.train.exponential_decay(INITIAL_LEARNING_RATE, global_step, decay_steps=1000, decay_rate=0.9)


        # Model.
        def model(data, dropout=False):
            # C-1
            convolution = tf.nn.conv2d(data, c_1_weights, [1, 1, 1, 1], padding='VALID')
            hidden = tf.nn.relu(convolution + c_1_biases)

            # S-2
            hidden = tf.nn.max_pool(value=hidden, ksize=[1, S_2_PATCH_SIZE, S_2_PATCH_SIZE, 1],
                                    strides=[1, S_2_PATCH_SIZE, S_2_PATCH_SIZE, 1], padding='VALID')

            # C-3
            convolution = tf.nn.conv2d(hidden, c_3_weights, [1, 1, 1, 1], padding='VALID')
            hidden = tf.nn.relu(convolution + c_3_biases)

            # S-4
            hidden = tf.nn.max_pool(value=hidden, ksize=[1, S_4_PATCH_SIZE, S_4_PATCH_SIZE, 1],
                                    strides=[1, S_4_PATCH_SIZE, S_4_PATCH_SIZE, 1], padding='VALID')
            if dropout:
                hidden = tf.nn.dropout(hidden, DROPOUT_PROBABILITY)

            # C-5
            shape = hidden.get_shape().as_list()
            reshape = tf.reshape(hidden, [shape[0], shape[1] * shape[2] * shape[3]])  # convert to 1-D
            hidden = tf.nn.relu(tf.matmul(reshape, c_5_weights) + c_5_biases)
            if dropout:
                hidden = tf.nn.dropout(hidden, DROPOUT_PROBABILITY)

            # F-6
            hidden = tf.nn.relu(tf.matmul(hidden, f_6_weights) + f_6_biases)
            if dropout:
                hidden = tf.nn.dropout(hidden, DROPOUT_PROBABILITY)

            # 4th (final softmax layer)
            return tf.matmul(hidden, output_weights) + output_biases


        # Training computation.
        logits = model(tf_train_dataset, dropout=True)
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits, tf_train_labels))

        # Optimizer.
        optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss=loss, global_step=global_step)

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
            feed_dict = {tf_train_dataset: batch_data, tf_train_labels: batch_labels}

            _, l, predictions = session.run([optimizer, loss, train_prediction], feed_dict=feed_dict)

            if step % 50 == 0:
                print('Minibatch loss at step %d: %f' % (step, l))
                print('Minibatch accuracy: %.1f%%' % accuracy(predictions, batch_labels))
                print('Validation accuracy: %.1f%%' % accuracy(valid_prediction.eval(), valid_labels))

        print('Test accuracy: %.1f%%' % accuracy(test_prediction.eval(), test_labels))
