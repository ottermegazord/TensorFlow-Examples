
import tensorflow as tf
import time

# Import MNIST data from tensorflow

from tensorflow.examples.tutorials.mnist import input_data
# mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

# Parameters
learning_rate = 0.5
training_epochs = 10000
batch_size = 128
display_step = 1

# Define tf input placeholders
x = tf.placeholder(tf.float32, [None, 784]) # (28, 28) None => dimensions can be any length
y = tf.placeholder(tf.float32, [None, 10]) # 0-9 arabic numerals one-hot output vector with digit labels

# set weight and bias parameters

W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([1,10]))

# Set model

#Softmax useful for predicting what is what
#1. Change input to logits/linear model
#2. Feed logit to softmaxc (softmax normalizes the output between a value of 0 to 1)
# convert evidence tallies into our predicted probabilities
# x,w is a trick because x is a 2d tensor
logits = tf.matmul(x, W) + b

# Cross entropy error minimization / not symmetric D(S,L) is not D(L,S) = -E
# H(p,q) = p(x)*log(q(x))
# one-hot vector [0,0,0,0,1,0,0,0]
#reduction indices => adds in second dimension
entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y)
cost = tf.reduce_mean(entropy)
# cost = tf.reduce_mean(-tf.reduce_sum(y*tf.log(pred), reduction_indices=1))
# cost = tf.nn.softmax_cross_entropy_with_logits(pred)
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost) # Repeat this to train model

#Initialize variables

init = tf.global_variables_initializer()

sess = tf.InteractiveSession()
init.run()

for epoch in range(training_epochs):
    batch = mnist.train.next_batch(batch_size)
    optimizer.run(feed_dict={x: batch[0], y: batch[1]})

correct_prediction = tf.equal(tf.argmax(logits,1), tf.argmax(y, 1))

accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32)) # cast convert to float

print(accuracy.eval(feed_dict={x: mnist.test.images, y: mnist.test.labels}))