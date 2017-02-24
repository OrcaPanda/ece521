import tensorflow as tf
import numpy as np

def euclid_distance(X, Z):
	X = (tf.expand_dims(X, 1))
	Z = (tf.expand_dims(Z, 0))
	return tf.reduce_sum(tf.squared_difference(X, Z), 2)

X = tf.Variable([[1,2],[3,2],[4,4]])
Z = tf.Variable([[4,2],[1,5],[4,7]])

Y = euclid_distance(X, Z)



sess = tf.Session()
sess.run(tf.global_variables_initializer())
print sess.run(Y)
