# -*- coding: UTF-8 -*-
"""
Python code for the assignment 2.
"""
from __future__ import print_function
import numpy as np
import tensorflow as tf
import cPickle as pickle

__author__ = 'kensk8er'

PICKLE_FILE = 'notMNIST.pickle'
IMAGE_SIZE = 28
NUM_LABELS = 10
BATCH_SIZE = 128
NUM_STEPS = 20001
HIDDEN_LAYER_1 = 1024
HIDDEN_LAYER_2 = 300
INITIAL_LEARNING_RATE = 0.04
LAMBDA_1 = 0.01
LAMBDA_2 = 0.01
LAMBDA_3 = 0.0001
DROPOUT_PROBABILITY = 0.5
EARLY_STOP_PERIOD = 30


def reformat(dataset, labels):
    dataset = dataset.reshape((-1, IMAGE_SIZE * IMAGE_SIZE)).astype(np.float32)
    # Map 0 to [1.0, 0.0, 0.0 ...], 1 to [0.0, 1.0, 0.0 ...]
    labels = (np.arange(NUM_LABELS) == labels[:, None]).astype(np.float32)
    return dataset, labels


def load_data():
    with open(PICKLE_FILE, 'rb') as f:
        save = pickle.load(f)
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


def accuracy(predictions, labels):
    return 100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1)) / predictions.shape[0]


if __name__ == '__main__':
    # load data
    train_dataset, train_labels, valid_dataset, valid_labels, test_dataset, test_labels = load_data()

    # reformat data
    train_dataset, train_labels = reformat(train_dataset, train_labels)
    valid_dataset, valid_labels = reformat(valid_dataset, valid_labels)
    test_dataset, test_labels = reformat(test_dataset, test_labels)
    print('Training set', train_dataset.shape, train_labels.shape)
    print('Validation set', valid_dataset.shape, valid_labels.shape)
    print('Test set', test_dataset.shape, test_labels.shape)

    # # restrict training data to just a few batches
    # train_dataset = train_dataset[:256]
    # train_labels = train_labels[:256]

    # Build the graph
    graph = tf.Graph()
    with graph.as_default():
        # Input data. For the training data, we use a placeholder that will be fed
        # at run time with a training minibatch.
        tf_train_dataset = tf.placeholder(tf.float32, shape=(BATCH_SIZE, IMAGE_SIZE * IMAGE_SIZE))
        tf_train_labels = tf.placeholder(tf.float32, shape=(BATCH_SIZE, NUM_LABELS))
        tf_valid_dataset = tf.constant(valid_dataset)
        tf_test_dataset = tf.constant(test_dataset)

        # Variables.
        weights_1 = tf.Variable(tf.truncated_normal([IMAGE_SIZE * IMAGE_SIZE, HIDDEN_LAYER_1]))
        biases_1 = tf.Variable(tf.zeros([HIDDEN_LAYER_1]))
        weights_2 = tf.Variable(tf.truncated_normal([HIDDEN_LAYER_1, HIDDEN_LAYER_2]))
        biases_2 = tf.Variable(tf.zeros([HIDDEN_LAYER_2]))
        weights_3 = tf.Variable(tf.truncated_normal([HIDDEN_LAYER_2, NUM_LABELS]))
        biases_3 = tf.Variable(tf.zeros([NUM_LABELS]))
        global_step = tf.Variable(0, trainable=False)
        learning_rate = tf.train.exponential_decay(INITIAL_LEARNING_RATE, global_step, decay_steps=1000, decay_rate=0.9)

        # Training computation.
        hidden_activations_1 = tf.nn.relu_layer(tf_train_dataset, weights_1, biases_1)
        # hidden_activations_1 = tf.nn.dropout(hidden_activations_1, DROPOUT_PROBABILITY)
        hidden_activations_2 = tf.nn.relu_layer(hidden_activations_1, weights_2, biases_2)
        # hidden_activations_2 = tf.nn.dropout(hidden_activations_2, DROPOUT_PROBABILITY)
        logits = tf.matmul(hidden_activations_2, weights_3) + biases_3
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits, tf_train_labels)) + \
               LAMBDA_1 * tf.nn.l2_loss(weights_1) + LAMBDA_2 * tf.nn.l2_loss(weights_2) + LAMBDA_3 * tf.nn.l2_loss(weights_3) + \
               LAMBDA_1 * tf.nn.l2_loss(biases_1) + LAMBDA_2 * tf.nn.l2_loss(biases_2) + LAMBDA_3 * tf.nn.l2_loss(weights_3) \

        # Optimizer.
        optimizer = tf.train.GradientDescentOptimizer(INITIAL_LEARNING_RATE).minimize(loss=loss,
                                                                                      global_step=global_step)

        # Predictions for the train, validation, and test data.
        train_prediction = tf.nn.softmax(
            tf.matmul(
                tf.nn.relu_layer(
                    tf.nn.relu_layer(tf_train_dataset, weights_1, biases_1),
                    weights_2, biases_2,
                ),
                weights_3
            ) + biases_3
        )
        valid_prediction = tf.nn.softmax(
            tf.matmul(
                tf.nn.relu_layer(
                    tf.nn.relu_layer(tf_valid_dataset, weights_1, biases_1),
                    weights_2, biases_2,
                ),
                weights_3
            ) + biases_3
        )
        test_prediction = tf.nn.softmax(
            tf.matmul(
                tf.nn.relu_layer(
                    tf.nn.relu_layer(tf_test_dataset, weights_1, biases_1),
                    weights_2, biases_2,
                ),
                weights_3
            ) + biases_3
        )

    # for early stopping
    past_valid_accuracy = [0. for _ in range(30)]
    average_valid_accuracy = 0.

    # Run the graph session
    with tf.Session(graph=graph) as session:
        tf.initialize_all_variables().run()
        print("Initialized")

        for step in range(NUM_STEPS):
            # Pick an offset within the training data, which has been randomized.
            # Note: we could use better randomization across epochs.
            offset = (step * BATCH_SIZE) % (train_labels.shape[0] - BATCH_SIZE)

            # Generate a minibatch.
            batch_data = train_dataset[offset:(offset + BATCH_SIZE), :]
            batch_labels = train_labels[offset:(offset + BATCH_SIZE), :]

            # Prepare a dictionary telling the session where to feed the minibatch.
            # The key of the dictionary is the placeholder node of the graph to be fed,
            # and the value is the numpy array to feed to it.
            feed_dict = {tf_train_dataset: batch_data, tf_train_labels: batch_labels}
            _, l, predictions = session.run([optimizer, loss, train_prediction], feed_dict=feed_dict)

            if step % 100 == 0:
                valid_accuracy = accuracy(valid_prediction.eval(), valid_labels)
                print("Minibatch loss at step %d: %f" % (step, l))
                print("Minibatch accuracy: %.1f%%" % accuracy(predictions, batch_labels))
                print("Validation accuracy: %.1f%%" % valid_accuracy)

                past_valid_accuracy.insert(0, valid_accuracy)

                if np.mean(past_valid_accuracy[:EARLY_STOP_PERIOD]) < average_valid_accuracy:
                    print('Early Stop.')
                    break
                else:
                    average_valid_accuracy = np.mean(past_valid_accuracy[:EARLY_STOP_PERIOD])

        print("Test accuracy: %.1f%%" % accuracy(test_prediction.eval(), test_labels))
