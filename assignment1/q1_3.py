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

#X = tf.Variable([[1,2],[3,2],[4,4]])
#Z = tf.Variable([[4,2],[1,5],[4,7]])

#Y = euclid_distance(X, Z)

def get_responsibilities(D, dims, k):
	#Get the indices of the shortest distances
	values, indices = tf.nn.top_k(tf.multiply(D,-1), k)
	#and reshape them into a 1D tensor
	indices = tf.reshape(indices, [dims[0]*k, ])
	#get indices for where to update the responsibility vectors
	I = tf.reshape(tf.add(tf.expand_dims(tf.range(0, dims[0]),1), tf.zeros([dims[0], k], tf.int32)), [dims[0] * k, ])
	#convert the indices to int64
	update_indices = tf.to_int64(tf.stack([I, indices], 1))
	#The responsibility vectors have 1/k for each value
	values = tf.add(tf.zeros([dims[0] * k, ]), 1.0/k)
	return tf.SparseTensor(update_indices, values, dims)


#Y = get_responsibilities(Y, [3,3], 2)

#Placeholders for data
trainD = tf.placeholder(tf.float64, [None])
trainT = tf.placeholder(tf.float64, [None])
comparisonD = tf.placeholder(tf.float64, [None])
comparisonT = tf.placeholder(tf.float64, [None])

sess = tf.Session()
sess.run(tf.global_variables_initializer())

# Question 1.3.2
k = 1
D_train = euclid_distance(trainD, comparisonD)
R_train = get_responsibilities(D_train, [trainD.get_shape[0], comparisonD.get_shape[0]],k) 
P_train = tf.sparse_tensor_dense_matmul(R_train, trainD)
Y = tf.reduce_sum(tf.pow(tf.subtract(P_train, comparisonT),2)) / (2*trainData.size)


print sess.run(Y)
