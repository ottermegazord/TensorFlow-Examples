import tensorflow as tf

a = tf.constant(2)
b = tf.constant(3)

#launch graph

with tf.Session() as sess:
    print("a: %i" % sess.run(a))
    print("b: %i" % sess.run(b))
    print("addition: %i" % sess.run(a+b))
    print("multiple: %i" % sess.run(a * b))


# Variable as graph input

a = tf.placeholder(tf.int16)
b = tf.placeholder(tf.int16)

# define operations

add = tf.add(a, b)
mul = tf.multiply(a, b)

# Launch graph

feed_dict = {a: 2, b: 3}
with tf.Session() as sess:
    print("addition output: %i" % sess.run(add, feed_dict))
    print("multiplication output: %i" % sess.run(mul, feed_dict))


matrix1 = tf.constant([[3., 3.]])
matrix2 = tf.constant([[2.], [2.]])

product = tf.matmul(matrix1, matrix2)

print(product)

with tf.Session() as sess:
    print("Output: %i" % sess.run(matrix1))