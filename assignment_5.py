# -*- coding: UTF-8 -*-
"""
Script for assignment 5.
"""
from __future__ import print_function

from urllib import urlretrieve
import collections
import math
import numpy as np
import os
import random
import tensorflow as tf
import zipfile
from matplotlib import pylab
from sklearn.manifold import TSNE
import cPickle as pickle

PROJECT_ROOT = '/Users/kensk8er/PycharmProjects/udacity'
EXPECTED_BYTES = 31344016
URL = 'http://mattmahoney.net/dc/'
VOCABULARY_SIZE = 50000
DATASET_FILE = os.path.join(PROJECT_ROOT, 'text8.pkl')

__author__ = 'kensk8er'

BATCH_SIZE = 128
EMBEDDING_DIMENSION = 128  # Dimension of the embedding vector.
SKIP_WINDOW_SIZE = 1  # How many words to consider left and right.

# We pick a random validation set to sample nearest neighbors. Here we limit the validation samples to the words that have a low numeric ID, which by
# construction are also the most frequent.
VALID_SIZE = 16  # Random set of words to evaluate similarity on.
VALID_WINDOW = 100  # Only pick dev samples in the head of the distribution.
NUM_SAMPLED = 64  # Number of negative examples to sample.

NUM_STEPS = 100001

# the number of words to show on the 2d visualization
NUM_POINTS = 400


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
            return tf.compat.as_str(zip_file.read(name)).split()


def build_dataset(words):
    count = [['UNK', -1]]  # UNK = UnKNown, -1 will be replaced by the actual count later
    count.extend(collections.Counter(words).most_common(VOCABULARY_SIZE - 1))
    dictionary = dict()

    for word, _ in count:
        dictionary[word] = len(dictionary)  # mapping from word to its index

    data = list()
    unk_count = 0

    for word in words:
        if word in dictionary:
            index = dictionary[word]
        else:
            index = 0  # dictionary['UNK']
            unk_count += 1

        data.append(index)

    count[0][1] = unk_count
    reverse_dictionary = dict(zip(dictionary.values(), dictionary.keys()))  # mapping from id to word

    return data, count, dictionary, reverse_dictionary


def generate_batch(batch_size, skip_window):
    global data_index

    batch = np.ndarray(shape=[batch_size, skip_window * 2], dtype=np.int32)
    labels = np.ndarray(shape=[batch_size, 1], dtype=np.int32)
    span = 2 * skip_window + 1  # [ skip_window target skip_window ]
    buffer = collections.deque(maxlen=span)

    # generate the initial state of buffer and date_index
    for _ in range(span):
        buffer.append(data[data_index])
        data_index = (data_index + 1) % len(data)

    # iterate over a batch
    for i in range(batch_size):
        batch[i] = [word_id for buffer_id, word_id in enumerate(buffer) if buffer_id != skip_window]
        labels[i, 0] = buffer[skip_window]
        buffer.append(data[data_index])
        data_index = (data_index + 1) % len(data)

    return batch, labels


def load_dataset(filename):
    if os.path.isfile(DATASET_FILE):
        with open(DATASET_FILE, 'rb') as dataset_file:
            dataset = pickle.load(dataset_file)
    else:
        # load words
        words = read_data(filename)
        print('Data size %d' % len(words))

        # get the count
        dataset = build_dataset(words)

        # Hint to reduce memory.
        del words

        with open(DATASET_FILE, 'wb') as dataset_file:
            pickle.dump(dataset, dataset_file, pickle.HIGHEST_PROTOCOL)

    data, count, dictionary, reverse_dictionary = dataset

    # print data statistics
    print('Most common words (+UNK)', count[:5])
    print('Sample data', data[:10])
    print('Data size: {}'.format(len(data)))

    return data, count, dictionary, reverse_dictionary


def plot(embeddings, labels):
    assert embeddings.shape[0] >= len(labels), 'More labels than embeddings'

    pylab.figure(figsize=(15, 15))  # in inches

    for i, label in enumerate(labels):
        x, y = embeddings[i, :]
        pylab.scatter(x, y)
        pylab.annotate(label, xy=(x, y), xytext=(5, 2), textcoords='offset points', ha='right', va='bottom')

    pylab.show()


if __name__ == '__main__':
    # prepare data
    filename = maybe_download('text8.zip', EXPECTED_BYTES)
    data, count, dictionary, reverse_dictionary = load_dataset(filename)
    print('data:', [reverse_dictionary[di] for di in data[:8]])

    # # define a graph
    graph = tf.Graph()
    valid_examples = np.array(random.sample(xrange(VALID_WINDOW), VALID_SIZE))

    with graph.as_default():
        # Input data.
        train_dataset = tf.placeholder(tf.int32, shape=[BATCH_SIZE, SKIP_WINDOW_SIZE * 2])
        train_labels = tf.placeholder(tf.int32, shape=[BATCH_SIZE, 1])
        valid_dataset = tf.constant(valid_examples, dtype=tf.int32)

        # Variables.
        embeddings = tf.Variable(tf.random_uniform([VOCABULARY_SIZE, EMBEDDING_DIMENSION], minval=-1.0, maxval=1.0))
        softmax_weights = tf.Variable(tf.truncated_normal([VOCABULARY_SIZE, EMBEDDING_DIMENSION * SKIP_WINDOW_SIZE * 2],
                                                          stddev=1.0 / math.sqrt(EMBEDDING_DIMENSION)))
        softmax_biases = tf.Variable(tf.zeros([VOCABULARY_SIZE]))

        # Model.
        # Look up embeddings for inputs.
        embed = tf.reshape(tf.nn.embedding_lookup(embeddings, train_dataset), shape=[BATCH_SIZE, -1])

        # Compute the softmax loss, using a sample of the negative labels each time.
        loss = tf.reduce_mean(tf.nn.sampled_softmax_loss(softmax_weights, softmax_biases, embed, train_labels, NUM_SAMPLED, VOCABULARY_SIZE))

        # Optimizer.
        optimizer = tf.train.AdagradOptimizer(learning_rate=1.0).minimize(loss)

        # Compute the similarity between minibatch examples and all embeddings. We use the cosine distance:
        norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), reduction_indices=1, keep_dims=True))
        normalized_embeddings = embeddings / norm
        valid_embeddings = tf.nn.embedding_lookup(normalized_embeddings, valid_dataset)
        similarity = tf.matmul(valid_embeddings, tf.transpose(normalized_embeddings))

    # # Run the graph

    # initialize data_index
    data_index = 0

    with tf.Session(graph=graph) as session:
        tf.initialize_all_variables().run()
        print('Initialized')

        average_loss = 0.

        for step in range(NUM_STEPS):
            batch_data, batch_labels = generate_batch(BATCH_SIZE, SKIP_WINDOW_SIZE)
            feed_dict = {train_dataset: batch_data, train_labels: batch_labels}
            _, l = session.run([optimizer, loss], feed_dict=feed_dict)
            average_loss += l

            if step % 2000 == 0:
                if step > 0:
                    average_loss /= 2000

                # The average loss is an estimate of the loss over the last 2000 batches.
                print('Average loss at step %d: %f' % (step, average_loss))
                average_loss = 0.

            # note that this is expensive (~20% slowdown if computed every 500 steps)
            if step % 10000 == 0:
                sim = similarity.eval()

                for i in xrange(VALID_SIZE):
                    valid_word = reverse_dictionary[valid_examples[i]]
                    top_k = 8  # number of nearest neighbors
                    nearest = (-sim[i, :]).argsort()[1: top_k + 1]  # 0-th element is itself?
                    log = 'Nearest to %s:' % valid_word

                    for k in xrange(top_k):
                        close_word = reverse_dictionary[nearest[k]]
                        log = '%s %s,' % (log, close_word)

                    print(log)

        final_embeddings = normalized_embeddings.eval()

    # reduce to 2-dimension
    tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000)
    two_d_embeddings = tsne.fit_transform(final_embeddings[1: NUM_POINTS + 1, :])

    # visualize the embeddings
    words = [reverse_dictionary[i] for i in range(1, NUM_POINTS + 1)]
    plot(two_d_embeddings, words)
