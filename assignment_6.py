# -*- coding: UTF-8 -*-
"""
Script for assignment 6.
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

VALID_SIZE = 1000

# model parameters
BATCH_SIZE = 64
NUM_UNROLLINGS = 10
NUM_NODES = 128
NUM_STEPS = 30001
SUMMARY_FREQUENCY = 100
EMBEDDING_DIMENSION = 128
DROPOUT_PROBABILITY = 0.5

CHARACTER_SIZE = (len(string.ascii_lowercase) + 1)  # [a-z] + ' '
VOCABULARY_SIZE = CHARACTER_SIZE ** 2  # [a-z] + ' ' (bigram)
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


def bigram2id(bigram):
    """easily extensible to ngram2id actually"""
    assert len(bigram) == 2, 'Input needs to be 2 characters.'
    char_id = 0

    for digit, char in enumerate(bigram):
        char_id += char2id(char) * (CHARACTER_SIZE ** digit)

    return char_id


def id2char(char_id):
    if char_id > 0:
        return chr(char_id + FIRST_LETTER - 1)
    else:
        return ' '


def id2bigram(char_id):
    first_digit_id = char_id % CHARACTER_SIZE
    second_digit_id = char_id // CHARACTER_SIZE

    return id2char(first_digit_id) + id2char(second_digit_id)


class BatchGenerator(object):
    def __init__(self, text, batch_size, num_unrollings):
        self._text = text
        self._text_size = len(text)
        self._batch_size = batch_size
        self._num_unrollings = num_unrollings
        segment = self._text_size // batch_size
        self._cursor = [offset * segment for offset in range(batch_size)]
        self._last_batch = self._next_batch()

    def _next_batch(self):
        """Generate a single batch from the current cursor position in the data."""
        batch = np.zeros(shape=(self._batch_size, VOCABULARY_SIZE), dtype=np.float)

        for batch_id in range(self._batch_size):
            batch[batch_id, bigram2id(self._text[self._cursor[batch_id]: self._cursor[batch_id] + 2])] = 1.0
            self._cursor[batch_id] = (self._cursor[batch_id] + 1) % (self._text_size - 1)

        return batch

    def next(self):
        """Generate the next array of batches from the data. The array consists of
        the last batch of the previous array, followed by num_unrollings new ones.
        """
        batches = [self._last_batch]

        for step in range(self._num_unrollings):
            batches.append(self._next_batch())

        self._last_batch = batches[-1]

        return batches


def characters(probabilities):
    """Turn a 1-hot encoding or a probability distribution over the possible
    characters back into its (most likely) character representation."""
    return [id2char(char_id) for char_id in np.argmax(probabilities, 1)]


def bigrams(probabilities):
    """Turn a 1-hot encoding or a probability distribution over the possible
    bigrams back into its (most likely) bigram representation."""
    return [id2bigram(bigram_id) for bigram_id in np.argmax(probabilities, 1)]


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


def bigram_label2unigram_label(bigram_one_hot_encodings, batch_size=BATCH_SIZE):
    unigram_id_labels = np.where(bigram_one_hot_encodings == 1)[1] // CHARACTER_SIZE
    return np.array([[float(char_id == unigram_id_labels[batch_id]) for char_id in range(CHARACTER_SIZE)] for batch_id in range(batch_size)])


if __name__ == '__main__':
    filename = maybe_download('text8.zip', 31344016)
    text = read_data(filename)

    valid_text = text[:VALID_SIZE]
    train_text = text[VALID_SIZE:]
    train_size = len(train_text)

    train_batches = BatchGenerator(train_text, BATCH_SIZE, NUM_UNROLLINGS)
    valid_batches = BatchGenerator(valid_text, 1, 1)

    # simple LSTM Model
    graph = tf.Graph()
    with graph.as_default():
        # Parameters for input, forget, cell state, and output gates
        W_lstm = tf.Variable(tf.truncated_normal([EMBEDDING_DIMENSION + NUM_NODES, NUM_NODES * 4]))
        b_lstm = tf.Variable(tf.zeros([1, NUM_NODES * 4]))

        # Variables saving state across unrollings.
        previous_output = tf.Variable(tf.zeros([BATCH_SIZE, NUM_NODES]), trainable=False)
        previous_state = tf.Variable(tf.zeros([BATCH_SIZE, NUM_NODES]), trainable=False)

        # Classifier weights and biases.
        W = tf.Variable(tf.truncated_normal([NUM_NODES, CHARACTER_SIZE], -0.1, 0.1))
        b = tf.Variable(tf.zeros([CHARACTER_SIZE]))

        # embedding
        embeddings = tf.Variable(tf.random_uniform([VOCABULARY_SIZE, EMBEDDING_DIMENSION], minval=-1.0, maxval=1.0))

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
            train_X.append(tf.placeholder(tf.int32, shape=[BATCH_SIZE, 1]))
            train_labels.append(tf.placeholder(tf.float32, shape=[BATCH_SIZE, CHARACTER_SIZE]))

        # Unrolled LSTM loop.
        outputs = list()
        output = previous_output
        state = previous_state

        for X in train_X:
            embed = tf.reshape(tf.nn.embedding_lookup(embeddings, X), shape=[BATCH_SIZE, -1])
            output, state = lstm_cell(embed, output, state)
            outputs.append(output)

        # State saving across unrollings.
        with tf.control_dependencies([previous_output.assign(output), previous_state.assign(state)]):
            # Classifier.
            logits = tf.nn.xw_plus_b(tf.nn.dropout(tf.concat(0, outputs), DROPOUT_PROBABILITY), W, b)
            loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits, tf.concat(0, train_labels)))

        # Optimizer.
        global_step = tf.Variable(0)
        learning_rate = tf.train.exponential_decay(10.0, global_step, 5000, 0.1, staircase=True)
        optimizer = tf.train.GradientDescentOptimizer(learning_rate)
        gradients, v = zip(*optimizer.compute_gradients(loss))
        gradients, _ = tf.clip_by_global_norm(gradients, 1.25)
        optimizer = optimizer.apply_gradients(zip(gradients, v), global_step=global_step)

        # Predictions.
        train_prediction = tf.nn.softmax(logits)

        # Sampling and validation eval: batch 1, no unrolling.
        sample_input = tf.placeholder(tf.int32, shape=[1, 1])
        sample_embed = tf.reshape(tf.nn.embedding_lookup(embeddings, sample_input), shape=[1, -1])
        previous_sample_output = tf.Variable(tf.zeros([1, NUM_NODES]))
        previous_sample_state = tf.Variable(tf.zeros([1, NUM_NODES]))
        reset_sample_state = tf.group(previous_sample_output.assign(tf.zeros([1, NUM_NODES])), previous_sample_state.assign(tf.zeros([1, NUM_NODES])))
        sample_output, sample_state = lstm_cell(sample_embed, previous_sample_output, previous_sample_state)

        with tf.control_dependencies([previous_sample_output.assign(sample_output), previous_sample_state.assign(sample_state)]):
            sample_prediction = tf.nn.softmax(tf.nn.xw_plus_b(sample_output, W, b))


    # Run the model
    with tf.Session(graph=graph) as session:
        tf.initialize_all_variables().run()
        print('Initialized')
        mean_loss = 0

        for step in range(NUM_STEPS):
            batches = train_batches.next()
            feed_dict = dict()

            for batch_id in range(NUM_UNROLLINGS):
                feed_dict[train_X[batch_id]] = np.where(batches[batch_id] == 1)[1].reshape((-1, 1))
                feed_dict[train_labels[batch_id]] = bigram_label2unigram_label(batches[batch_id + 1])

            _, l, predictions, lr = session.run([optimizer, loss, train_prediction, learning_rate], feed_dict=feed_dict)

            mean_loss += l

            if step % SUMMARY_FREQUENCY == 0:
                if step > 0:
                    mean_loss = mean_loss / SUMMARY_FREQUENCY

                # The mean loss is an estimate of the loss over the last few batches.
                print('Average loss at step %d: %f learning rate: %f' % (step, mean_loss, lr))

                mean_loss = 0
                labels = np.concatenate([bigram_label2unigram_label(batch) for batch in batches[1:]])
                print('Minibatch perplexity: %.2f' % float(np.exp(log_prob(predictions, labels))))

                if step % (SUMMARY_FREQUENCY * 10) == 0:
                    # Generate some samples.
                    print('=' * 80)

                    for _ in range(5):
                        feed = sample(random_distribution())
                        sentence = bigrams(feed)[0]
                        reset_sample_state.run()

                        for _ in range(79):
                            feed = np.where(feed == 1)[1].reshape((-1, 1))
                            prediction = sample_prediction.eval({sample_input: feed})
                            feed = sample(prediction)
                            sentence += characters(feed)[0]
                            last_bigram_id = bigram2id(sentence[-2:])
                            feed = np.array([[float(last_bigram_id == bigram_id) for bigram_id in range(VOCABULARY_SIZE)]])

                        print(sentence)

                    print('=' * 80)

                # Measure validation set perplexity.
                reset_sample_state.run()
                valid_log_prob = 0

                for _ in range(VALID_SIZE):
                    valid_batch = valid_batches.next()
                    predictions = sample_prediction.eval({sample_input: np.where(valid_batch[0] == 1)[1].reshape((-1, 1))})
                    valid_log_prob = valid_log_prob + log_prob(predictions, bigram_label2unigram_label(valid_batch[1], batch_size=1))

                print('Validation set perplexity: %.2f' % float(np.exp(valid_log_prob / VALID_SIZE)))
