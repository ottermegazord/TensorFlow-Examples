import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv('FIFA18v2.csv', delimiter=',')
# print(df['Rating'].as_matrix())

train_X = np.asarray(df['Overall'].as_matrix())
train_Y = np.asarray(df['Value'].as_matrix())
learning_rate = 0.01
epochs = 5
display_step = 1

# train_X = np.asarray([3.3,4.4,5.5,6.71,6.93,4.168,9.779,6.182,7.59,2.167,
#                          7.042,10.791,5.313,7.997,5.654,9.27,3.1])
# train_Y = np.asarray([1.7,2.76,2.09,3.19,1.694,1.573,3.366,2.596,2.53,1.221,
#                          2.827,3.465,1.65,2.904,2.42,2.94,1.3])

print(train_X)

# Graph Input
X = tf.placeholder("float")
Y = tf.placeholder("float")

# Weights and Bias

W = tf.Variable(np.random.randn(), name='weight')
b = tf.Variable(np.random.randn(), name='bias')

# create linear regression model WX + b

pred = tf.add(tf.multiply(W, X), b)
n_samples = train_X.shape[0]

# cost function => MSE

cost = tf.reduce_sum(tf.pow(pred-Y, 2) / (2*n_samples))

# gradient descent

optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

# start training

# initial model

init = tf.global_variables_initializer()


with tf.Session() as sess:
    print("Initiate training")
    sess.run(init)

    for epoch in range(epochs):
        print("Training")
        for (x, y) in zip(train_X, train_Y):
            sess.run(optimizer, feed_dict={X: x, Y:y})

        # Display logs per epoch step
        if (epoch + 1) % display_step == 0:
            c = sess.run(cost, feed_dict={X: train_X, Y: train_Y})
            print "Epoch:", '%04d' % (epoch + 1), "cost=", "{:.9f}".format(c), \
                "W=", sess.run(W), "b=", sess.run(b)

    training_cost = sess.run(cost, feed_dict={X: train_X, Y: train_Y})

    print "Training cost=", training_cost, "W=", sess.run(W), "b=", sess.run(b), '\n'
    plt.plot(train_X, train_Y, 'ro', label='Original data')
    plt.plot(train_X, train_X*sess.run(W) + sess.run(b), label='regression')
    plt.show()



