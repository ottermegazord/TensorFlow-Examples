from __future__ import absolute_import, division, print_function

import tensorflow as tf
import tensorflow.contrib.eager as tfe
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import preprocessing

FILEPATH = 'FIFA18v2.csv'

# Load data

df = pd.read_csv(FILEPATH, delimiter=',')

train_X = np.asarray(df['Wage'].as_matrix())
train_Y = np.asarray(df['Overall'].as_matrix())

train_X = preprocessing.scale(train_X)
# X_test = preprocessing.scale(X_test)
train_Y = preprocessing.scale(train_Y)
# y_test = preprocessing.scale(y_test)

# train_X = np.array([3.3, 4.4, 5.5, 6.71, 6.93, 4.168, 9.779, 6.182, 7.59, 2.167,
#            7.042, 10.791, 5.313, 7.997, 5.654, 9.27, 3.1])
# train_Y = np.array([1.7, 2.76, 2.09, 3.19, 1.694, 1.573, 3.366, 2.596, 2.53, 1.221,
#            2.827, 3.465, 1.65, 2.904, 2.42, 2.94, 1.3])

# Set parameters

learning_rate = 0.01
epochs = 20
display_step = 10
num_steps = 2000
# n_samples = train_X.shape[0]

n_samples = train_X.shape[0]

# Create variables for weight and bias

W1 = tfe.Variable(np.random.randn())
W2 = tfe.Variable(np.random.randn())
# W3 = tfe.Variable(np.random.randn())
b = tfe.Variable(np.random.randn())

X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)

# pred = tf.add(tf.add(tf.multiply(W2, tf.multiply(X,X)),tf.multiply(W1,X)), b)
pred = W2 * np.power(train_X, 2) + W1 * train_X + b

cost = tf.reduce_sum(tf.pow(pred-Y, 2) / (2*n_samples))

optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)
# optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for epoch in range(epochs):
        print("Training")
        for (x, y) in zip(train_X, train_Y):
            sess.run(optimizer, feed_dict={X: x, Y:y})

        c = sess.run(cost, feed_dict={X: train_X, Y: train_Y})
        print("Epoch:", '%04d' % (epoch + 1), "cost=", "{:.9f}".format(c))


    w2, w1, B = sess.run([W2, W1, b])

    print (w1, w1, B)
    output = w2 * np.power(train_X, 2) + w1 * train_X + B
    print (output)
    plt.scatter(train_X, output, label='Fitted line', zorder=10)
    plt.plot(train_X, train_Y, 'ro', label='Original data', zorder=0)
    plt.legend()
    plt.show()