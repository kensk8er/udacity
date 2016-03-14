# -*- coding: UTF-8 -*-
"""
Script for assignment 6. question 3.
"""

from __future__ import print_function
import os
from urllib import urlretrieve
import numpy as np
import random
import string

import tensorflow as tf
import zipfile

__author__ = 'kensk8er'

PROJECT_ROOT = '/Users/kensk8er/PycharmProjects/udacity'
DATASET_FILE = os.path.join(PROJECT_ROOT, 'text8.pkl')
URL = 'http://mattmahoney.net/dc/'

VALID_SIZE = 100

# model parameters
BATCH_SIZE = 64
NUM_UNROLLINGS = 20
NUM_NODES = 128
NUM_STEPS = 30001
SUMMARY_FREQUENCY = 100
DROPOUT_PROBABILITY = 0.5

VOCABULARY_SIZE = (len(string.ascii_lowercase) + 1)  # [a-z] + ' '
FIRST_LETTER = ord(string.ascii_lowercase[0])


def maybe_download(filename, expected_bytes):
    """Download a file if not present, and make sure it's the right size."""
    if not os.path.exists(filename):
        filename, _ = urlretrieve(URL + filename, filename)

    stat_info = os.stat(filename)

    if stat_info.st_size == expected_bytes:
        print('Found and verified %s' % filename)
    else:
        print(stat_info.st_size)
        raise Exception('Failed to verify ' + filename + '. Can you get to it with a browser?')

    return filename


def read_data(filename):
    with zipfile.ZipFile(filename) as zip_file:
        for name in zip_file.namelist():
            return tf.compat.as_str(zip_file.read(name))


def char2id(char):
    if char in string.ascii_lowercase:
        return ord(char) - FIRST_LETTER + 1
    elif char == ' ':
        return 0
    else:
        print('Unexpected character: %s' % char)
        return 0


def id2char(char_id):
    if char_id > 0:
        return chr(char_id + FIRST_LETTER - 1)
    else:
        return ' '


def mirror_string(string):
    space = ' '
    previous_index = None
    space_index = None

    while True:
        try:
            space_index = string.index(space, 0 if previous_index is None else previous_index + 1)

            if previous_index is None:
                string = "{0} {1}".format(string[:space_index][::-1], string[space_index + 1:])
            else:
                string = "{0} {1} {2}".format(string[:previous_index], string[previous_index + 1: space_index][::-1], string[space_index + 1:])

            previous_index = space_index

        except ValueError:
            break

    if space_index:
        string = "{0} {1}".format(string[:space_index], string[space_index + 1:][::-1])

    return string


class BatchGenerator(object):
    def __init__(self, text, batch_size, num_unrollings):
        self._text = text
        self._text_size = len(text)
        self._batch_size = batch_size
        self._num_unrollings = num_unrollings
        segment = self._text_size // batch_size
        self._cursor = [offset * segment for offset in range(batch_size)]

    def _next_batch(self):
        """Generate a single batch from the current cursor position in the data."""
        batch = np.zeros(shape=(self._batch_size, VOCABULARY_SIZE), dtype=np.float)

        for batch_id in range(self._batch_size):
            batch[batch_id, char2id(self._text[self._cursor[batch_id]: self._cursor[batch_id] + 1])] = 1.0
            self._cursor[batch_id] = (self._cursor[batch_id] + 1) % self._text_size

        return batch

    def next(self):
        """Generate the next array of batches from the data. The array consists of
        the last batch of the previous array, followed by num_unrollings new ones.
        """
        batches = list()
        labels = list()

        for step in range(self._num_unrollings):
            batches.append(self._next_batch())

        batch_id2string = batches2string(batches)
        batch_id2string = [mirror_string(string) for string in batch_id2string]

        for step in range(self._num_unrollings):
            label = np.zeros(shape=(self._batch_size, VOCABULARY_SIZE), dtype=np.float)

            for batch_id in range(self._batch_size):
                label[batch_id, char2id(batch_id2string[batch_id][step])] = 1.0

            labels.append(label)

        return batches, labels


def characters(probabilities):
    """Turn a 1-hot encoding or a probability distribution over the possible
    characters back into its (most likely) character representation."""
    return [id2char(char_id) for char_id in np.argmax(probabilities, 1)]


def batches2string(batches):
    """Convert a sequence of batches back into their (most likely) string
    representation."""
    string = [''] * batches[0].shape[0]

    for batch in batches:
        string = [''.join(string_tuple) for string_tuple in zip(string, characters(batch))]

    return string


def log_prob(predictions, labels):
    """Log-probability of the true labels in a predicted batch."""
    predictions[predictions < 1e-10] = 1e-10
    return np.sum(np.multiply(labels, -np.log(predictions))) / labels.shape[0]


def sample_distribution(distribution):
    """Sample one element from a distribution assumed to be an array of normalized probabilities."""
    r = random.uniform(0, 1)
    s = 0

    for i in range(len(distribution)):
        s += distribution[i]

        if s >= r:
            return i

    return len(distribution) - 1


def sample(prediction):
    """Turn a (column) prediction into 1-hot encoded samples."""
    p = np.zeros(shape=[1, VOCABULARY_SIZE], dtype=np.float)
    p[0, sample_distribution(prediction[0])] = 1.0
    return p


def random_distribution():
    """Generate a random column of probabilities."""
    b = np.random.uniform(0.0, 1.0, size=[1, VOCABULARY_SIZE])
    return b / np.sum(b, 1)[:, None]


if __name__ == '__main__':
    filename = maybe_download('text8.zip', 31344016)
    text = read_data(filename)

    valid_text = text[:VALID_SIZE]
    train_text = text[VALID_SIZE:]
    train_size = len(train_text)

    train_batches = BatchGenerator(train_text, BATCH_SIZE, NUM_UNROLLINGS)
    valid_batches = BatchGenerator(valid_text, 1, NUM_UNROLLINGS)

    # simple LSTM Model
    graph = tf.Graph()
    with graph.as_default():
        # Parameters for input, forget, cell state, and output gates
        W_lstm = tf.Variable(tf.truncated_normal([VOCABULARY_SIZE + NUM_NODES, NUM_NODES * 4]))
        b_lstm = tf.Variable(tf.zeros([1, NUM_NODES * 4]))

        # Initial values for output, state and X
        initial_output = tf.constant(np.zeros([BATCH_SIZE, NUM_NODES]), dtype=tf.float32)
        initial_state = tf.constant(np.zeros([BATCH_SIZE, NUM_NODES]), dtype=tf.float32)
        X_initial = tf.constant(np.zeros([BATCH_SIZE, VOCABULARY_SIZE]), dtype=tf.float32)

        # Classifier weights and biases.
        W = tf.Variable(tf.truncated_normal([NUM_NODES, VOCABULARY_SIZE], -0.1, 0.1))
        b = tf.Variable(tf.zeros([VOCABULARY_SIZE]))

        # Definition of the cell computation.
        def lstm_cell(X, output, state):
            """Create a LSTM cell. See e.g.: http://arxiv.org/pdf/1402.1128v1.pdf
            Note that in this formulation, we omit the various connections between the
            previous state and the gates."""
            X_output = tf.concat(1, [X, output])
            all_logits = tf.matmul(X_output, W_lstm) + b_lstm

            input_gate = tf.sigmoid(all_logits[:, :NUM_NODES])
            forget_gate = tf.sigmoid(all_logits[:, NUM_NODES: NUM_NODES * 2])
            output_gate = tf.sigmoid(all_logits[:, NUM_NODES * 2: NUM_NODES * 3])
            temp_state = all_logits[:, NUM_NODES * 3:]
            state = forget_gate * state + input_gate * tf.tanh(temp_state)

            return output_gate * tf.tanh(state), state


        # Input data.
        train_X = list()
        train_labels = list()
        for _ in range(NUM_UNROLLINGS):
            train_X.append(tf.placeholder(tf.float32, shape=[BATCH_SIZE, VOCABULARY_SIZE]))
            train_labels.append(tf.placeholder(tf.float32, shape=[BATCH_SIZE, VOCABULARY_SIZE]))

        # Unrolled LSTM loop.
        output = initial_output
        state = initial_state

        # memorizing input sequence
        for train_x in train_X:
            output, state = lstm_cell(train_x, output, state)

        # predicting mirrored sequence
        train_predict_X = X_initial
        losses = list()
        train_predictions = list()
        for unroll_id in range(NUM_UNROLLINGS):
            output, state = lstm_cell(train_predict_X, output, state)
            train_logit = tf.nn.xw_plus_b(output, W, b)
            train_predict_X = tf.nn.softmax(train_logit)
            train_predictions.append(train_predict_X)

            loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(train_logit, train_labels[unroll_id]))
            losses.append(loss)

        train_predictions = tf.reshape(tf.concat(1, train_predictions), [-1, VOCABULARY_SIZE])

        # # Classifier.
        # logits = tf.nn.xw_plus_b(tf.nn.dropout(tf.concat(0, outputs), DROPOUT_PROBABILITY), W, b)
        # loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits, tf.concat(0, train_labels)))
        # loss = tf.concat(0, losses)
        loss = tf.add_n(losses)

        # Optimizer.
        global_step = tf.Variable(0)
        learning_rate = tf.train.exponential_decay(10.0, global_step, 5000, 0.1, staircase=True)
        optimizer = tf.train.GradientDescentOptimizer(learning_rate)
        gradients, v = zip(*optimizer.compute_gradients(loss))
        gradients, _ = tf.clip_by_global_norm(gradients, 1.25)
        optimizer = optimizer.apply_gradients(zip(gradients, v), global_step=global_step)

        # # Predictions.
        # train_prediction = tf.nn.softmax(logits)

        # Sampling and validation eval: batch 1, no unrolling.
        valid_X = list()
        valid_labels = list()
        for _ in range(NUM_UNROLLINGS):
            valid_X.append(tf.placeholder(tf.float32, shape=[1, VOCABULARY_SIZE]))
            valid_labels.append(tf.placeholder(tf.float32, shape=[1, VOCABULARY_SIZE]))

        initial_sample_output = tf.constant(np.zeros([1, NUM_NODES]), dtype=tf.float32)
        initial_sample_state = tf.constant(np.zeros([1, NUM_NODES]), dtype=tf.float32)

        sample_output = initial_sample_output
        sample_state = initial_sample_state

        # memorizing input sequence
        for valid_x in valid_X:
            sample_output, sample_state = lstm_cell(valid_x, sample_output, sample_state)

        # predicting mirrored sequence
        valid_initial_X = tf.constant(np.zeros([1, VOCABULARY_SIZE]), dtype=tf.float32)
        valid_predict_X = valid_initial_X
        sample_predictions = list()
        for unroll_id in range(NUM_UNROLLINGS):
            sample_output, sample_state = lstm_cell(valid_predict_X, sample_output, sample_state)
            sample_logit = tf.nn.xw_plus_b(sample_output, W, b)
            valid_predict_X = tf.nn.softmax(sample_logit)
            sample_predictions.append(valid_predict_X)

        sample_predictions = tf.reshape(tf.concat(1, sample_predictions), [-1, VOCABULARY_SIZE])

    # Run the model
    with tf.Session(graph=graph) as session:
        tf.initialize_all_variables().run()
        print('Initialized')
        mean_loss = 0

        for step in range(NUM_STEPS):
            batches, labels = train_batches.next()
            feed_dict = dict()

            for batch_id in range(NUM_UNROLLINGS):
                feed_dict[train_X[batch_id]] = batches[batch_id]
                feed_dict[train_labels[batch_id]] = labels[batch_id]

            _, l, predictions, lr = session.run([optimizer, loss, train_predictions, learning_rate], feed_dict=feed_dict)

            mean_loss += l

            if step % SUMMARY_FREQUENCY == 0:
                if step > 0:
                    mean_loss /= SUMMARY_FREQUENCY

                # The mean loss is an estimate of the loss over the last few batches.
                print('Average loss at step %d: %f learning rate: %f' % (step, mean_loss, lr))

                mean_loss = 0
                labels = np.concatenate(labels)
                print('Minibatch perplexity: %.2f' % float(np.exp(log_prob(predictions, labels))))

                if step % (SUMMARY_FREQUENCY * 10) == 0:
                    # Generate some samples.
                    print('=' * 80)

                    for _ in range(5):
                        batches, _ = valid_batches.next()
                        sentence = '"' + batches2string(batches)[0] + '" -> "'

                        predictions = sample_predictions.eval({valid_X[batch_id]: batches[batch_id] for batch_id in range(len(batches))})
                        sentence += ''.join(characters(predictions))
                        print(sentence + '"\n')

                    print('=' * 80)

                # Measure validation set perplexity.
                valid_log_prob = 0

                for _ in range(VALID_SIZE):
                    batches, labels = valid_batches.next()
                    predictions = sample_predictions.eval({valid_X[batch_id]: batches[batch_id] for batch_id in range(len(batches))})
                    labels = np.concatenate(labels)
                    valid_log_prob = valid_log_prob + log_prob(predictions, labels)

                print('Validation set perplexity: %.2f' % float(np.exp(valid_log_prob / VALID_SIZE)))
