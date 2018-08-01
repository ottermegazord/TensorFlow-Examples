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

# train_X = [3.3, 4.4, 5.5, 6.71, 6.93, 4.168, 9.779, 6.182, 7.59, 2.167,
#            7.042, 10.791, 5.313, 7.997, 5.654, 9.27, 3.1]
# train_Y = [1.7, 2.76, 2.09, 3.19, 1.694, 1.573, 3.366, 2.596, 2.53, 1.221,
#            2.827, 3.465, 1.65, 2.904, 2.42, 2.94, 1.3]

# Set parameters

learning_rate = 0.01
epochs = 5
display_step = 100
num_steps = 2000
# n_samples = train_X.shape[0]

n_samples = len(train_X)

# Set Eager API

tfe.enable_eager_execution()

# Create variables for weight and bias

W = tfe.Variable(np.random.randn())
b = tfe.Variable(np.random.randn())


# Create Linear Regression function
def linear_regression(theta):
    return theta*W + b


# Create MSE function (y-yi)^2 / 2*n_samples
def MSE(model, theta, labels):
    #return tf.reduce_sum(tf.pow(model(theta) - labels, 2) / (2*n_samples))
    return tf.reduce_mean(tf.square(tf.subtract(model(theta), labels)))

# Define SGD Optimizer
optimizer = tf.train.GradientDescentOptimizer(learning_rate)


# Computation of gradients
grad = tfe.implicit_gradients(MSE)

# Print initial cost
cost = MSE(linear_regression, train_X, train_Y)
print("Cost: %.9f" % cost)
print("Weight: %s" % W.numpy())
print("Bias: %s" % b.numpy())

for step in range(num_steps):

    optimizer.apply_gradients(grad(linear_regression, train_X, train_Y))

    if (step + 1) % display_step == 0 or step == 0:
        print("Epoch:", '%04d' % (step + 1), "cost=",
              "{:.9f}".format(MSE(linear_regression, train_X, train_Y)),
              "W=", W.numpy(), "b=", b.numpy())

plt.plot(train_X, train_Y, 'ro', label='Original data')
plt.plot(train_X, np.array(W * train_X + b), label='Fitted line')
plt.legend()
plt.show()