import tensorflow as tf
import numpy as np
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


size_of_set = 10

# Code given in the assignment
np.random.seed(521)
Data = np.linspace(1.0 , 10.0 , num =100) [:, np. newaxis]
Target = np.sin( Data ) + 0.1 * np.power( Data , 2) + 0.5 * np.random.randn(100 , 1)
randIdx = np.arange(100)
np.random.shuffle(randIdx)
trainData, trainTarget = Data[randIdx[:80]], Target[randIdx[:80]]
validData, validTarget = Data[randIdx[80:90]], Target[randIdx[80:90]]
testData, testTarget = Data[randIdx[90:100]], Target[randIdx[90:100]]

#trainData = np.linspace(1.0, 10.0, num = 80)[:, np. newaxis]
#trainTarget = np.linspace(1.0, 20.0, num = 80)[:, np. newaxis]


def euclid_distance(X, Z):
	X = (tf.expand_dims(X, 1))
	Z = (tf.expand_dims(Z, 0))
	return tf.reduce_sum(tf.squared_difference(X, Z), 2)

def get_responsibilities(D, dims, k):
	#Get the indices of the shortest distances
	values, indices = tf.nn.top_k(tf.transpose(tf.multiply(D,-1)), k)
	#print indices.get_shape()
	#and reshape them into a 1D tensor
	indices = tf.reshape(indices, [dims[0]*k, ])
	#get indices for where to update the responsibility vectors
	I = tf.reshape(tf.add(tf.expand_dims(tf.range(0, dims[0]),1), tf.zeros([dims[0], k], tf.int32)), [dims[0] * k, ])
	#convert the indices to int64
	update_indices = tf.to_int64(tf.stack([I, indices], 1))
	#The responsibility vectors have 1/k for each value
	values = tf.add(tf.zeros([dims[0] * k, ]), 1.0/k)
	return tf.sparse_to_dense(update_indices, dims, values, validate_indices = False)

def f(k, comparisonData, comparisonTarget, size_of_set, returnPredictions=False):

	#Placeholders for data
	X = tf.placeholder(tf.float64, [80,1])
	X_T = tf.placeholder(tf.float64, [size_of_set,1])
	Y = tf.placeholder(tf.float64, [80,1])
	Y_T = tf.placeholder(tf.float64, [size_of_set,1])
	sess = tf.Session()
	sess.run(tf.global_variables_initializer())
	
	# Question 1.3.2
	D_train = euclid_distance(X, X_T)
	#print D_train.get_shape()
	R_train = tf.cast(get_responsibilities(D_train, [size_of_set,80], k), tf.float64) 
	#print R_train.get_shape(), trainD.get_shape()
	#print R_train.get_shape(), Y
	P_train = tf.transpose(tf.matmul(R_train, Y))

	MSE = tf.reduce_sum(tf.pow(tf.subtract(P_train, tf.transpose(Y_T)),2)) / (2*size_of_set)
	if (not returnPredictions):
		print sess.run(MSE, feed_dict = {X:trainData, X_T: comparisonData, Y: trainTarget, Y_T: comparisonTarget})
	else:
		return sess.run(P_train, feed_dict = {X:trainData, X_T: comparisonData, Y: trainTarget, Y_T: comparisonTarget})
def notplot():
	for t in [[testData, testTarget, "test",10], [validData, validTarget, "valid",10], [trainData, trainTarget, "train",80]]:
		print t[2]

		for k in [1,3,5,50]:
	
			xx = [ x-np.average(trainTarget) for x in validTarget]
	
			s = np.sum(np.power(xx,2))/20.0
			print "k =", k
			f(k, t[0], t[1], t[3])
			#print s

def plot():
	x = np.linspace(0.0, 11.0, num = 1000)[:,np.newaxis]
	for k in [1,3,5,50]:
		y = f(k, x, x, x.size, True)
		plt.figure()
		plt.plot(x,np.transpose(y))
		
	plt.show()
		

plot()




