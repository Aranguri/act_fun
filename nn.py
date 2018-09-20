import tensorflow as tf
from tensorflow import keras
import numpy as np
import itertools
import time
import matplotlib.pyplot as plt

learning_rate = 1e-3

imdb = keras.datasets.imdb
word_index = imdb.get_word_index()

word_index = {k:(v+3) for k,v in word_index.items()}
word_index["<PAD>"] = 0
word_index["<START>"] = 1
word_index["<UNK>"] = 2  # unknown
word_index["<UNUSED>"] = 3

reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])

def decode_review(text):
    return ' '.join([reverse_word_index.get(i, '?') for i in text])

(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=10000)
train_data = keras.preprocessing.sequence.pad_sequences(train_data,
                                                        value=word_index["<PAD>"],
                                                        padding='post',
                                                        maxlen=256)

test_data = keras.preprocessing.sequence.pad_sequences(test_data,
                                                       value=word_index["<PAD>"],
                                                       padding='post',
                                                       maxlen=256)

def psh(arrays):
    for array in arrays: print(np.shape(array))

def normalize(array):
    return (array - array.mean()) / array.std()

batch_size = 100

def run():
    tf.reset_default_graph()
    act_1 = tf.nn.swish
    act_2 = tf.nn.swish
    x = tf.placeholder(tf.int32, [batch_size, 256])
    y = tf.placeholder(tf.int32, [batch_size,])
    embeddings = tf.Variable(tf.random_normal((10000, 64), stddev=.1))
    input_vectors = tf.nn.embedding_lookup(embeddings, x)
    input_vectors = tf.reduce_sum(input_vectors, axis=1)
    w1 = tf.Variable(tf.random_normal((64, 100)))
    b1 = tf.Variable(tf.zeros((1)))
    r = tf.Variable(.5)#shape=[1], initializer=tf.constant_initializer(.5), name='r')
    r_clip = tf.clip_by_value(r, 0, 1)
    #r_clip = tf.maximum(tf.minimum(1.0, r), 0.0)
    scores = tf.matmul(input_vectors, w1) + b1
    scores = r_clip * act_1(scores) + (1 - r_clip) * act_2(scores)
    w2 = tf.Variable(tf.random_normal((100, 2)))
    b2 = tf.Variable(tf.zeros((2)))
    final = tf.matmul(scores, w2) + b2
    final = act_1(final) + act_2(final)
    losses = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=final)
    loss = tf.reduce_mean(losses)
    params = [w1, b1, w2, b2, r]
    grad_params = tf.gradients(loss, params)
    new_ws = [tf.assign_sub(param, learning_rate * grad)
              for param, grad in zip(params, grad_params)]

    with tf.control_dependencies(new_ws):
        loss = tf.identity(loss)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for j in itertools.count():
            train_loss, test_loss = [], []
            p = np.random.permutation(len(train_data))
            new_train_data, new_train_labels = train_data[p], train_labels[p]
            new_train_data = new_train_data.reshape(-1, batch_size, 256)
            new_train_labels = new_train_labels.reshape(-1, batch_size)
            for i, (x_np, y_np) in enumerate(zip(new_train_data, new_train_labels)):
                train_loss.append(sess.run(loss, feed_dict={x: x_np, y: y_np}))

            new_test_data, new_test_labels = test_data[p], test_labels[p]
            new_test_data = new_test_data.reshape(-1, batch_size, 256)
            new_test_labels = new_test_labels.reshape(-1, batch_size)
            for i, (x_np, y_np) in enumerate(zip(new_test_data, new_test_labels)):
                test_loss.append(sess.run(loss, feed_dict={x: x_np, y: y_np}))
            print (j, np.mean(test_loss))

run()
