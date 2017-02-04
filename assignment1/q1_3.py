import tensorflow as tf
import numpy as np

# Code given in the assignment
np.random.seed(521)
Data = np.linspace(1.0 , 10.0 , num =100) [:, np. newaxis]
Target = np.sin( Data ) + 0.1 * np.power( Data , 2) + 0.5 * np.random.randn(100 , 1)
randIdx = np.arange(100)
np.random.shuffle(randIdx)
trainData, trainTarget = Data[randIdx[:80]], Target[randIdx[:80]]
validData, validTarget = Data[randIdx[80:90]], Target[randIdx[80:90]]
testData, testTarget = Data[randIdx[90:100]], Target[randIdx[90:100]]


def euclid_distance(X, Z):
	X = (tf.expand_dims(X, 1))
	Z = (tf.expand_dims(Z, 0))
	return tf.reduce_sum(tf.squared_difference(X, Z), 2)

X = tf.Variable([[1,2],[3,2],[4,4]])
Z = tf.Variable([[4,2],[1,5],[4,7]])

Y = euclid_distance(X, Z)

def get_responsibilities(D, dims, k):
	indices = tf.constant([[0], [2]])
	updates = tf.constant([8,8,8,8])
	shape = tf.constant([4, 4])

	values, indices = tf.nn.top_k(tf.multiply(D,-1), k)

	return tf.transpose(tf.reshape(tf.tile(tf.range(0,dims[0]),[dims[1]]),dims))
	
	#indices = tf.constant([[0],[1]])
	#updates = tf.constant(1/k)
	#shape = tf.constant(dims)
	#return tf.scatter_nd(indices, updates, shape)


R = get_responsibilities(Y, [3,3], 1)

sess = tf.Session()
sess.run(tf.global_variables_initializer())
print sess.run(Y)
print sess.run(R)
