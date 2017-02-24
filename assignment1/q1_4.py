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

#Kxx
def exp_kernel(D, lam):
	
	return tf.exp( tf.scalar_mul(-lam, D) ) 

def soft_knn(kx):
	
	return tf.truediv( kx, tf.reduce_sum(kx, 0) )
	#return tf.scalar_mul( 1.0/tf.reduce_sum(kx,0), kx  )

def gauss_reg(kXx, kxx):
	return tf.matmul( tf.matrix_inverse(kXx), kxx)

def f(k, comparisonData, comparisonTarget, size_of_set, returnPredictions=False):

	#Placeholders for data
	X = tf.placeholder(tf.float64, [80,1])
	X_T = tf.placeholder(tf.float64, [size_of_set,1])
	Y = tf.placeholder(tf.float64, [80,1])
	Y_T = tf.placeholder(tf.float64, [size_of_set,1])
	sess = tf.Session()
	sess.run(tf.global_variables_initializer())
	
	lam = 100

	# soft kern
	#print X.get_shape(), X_T.get_shape()
	D_train = euclid_distance(X, X_T)
	print D_train.get_shape()
	kernel = exp_kernel(D_train, lam)
	print kernel.get_shape()
	R_train_soft = soft_knn(kernel)
	print R_train_soft.get_shape(), Y.get_shape()
	P_train_soft = tf.transpose(tf.matmul(tf.transpose(R_train_soft), Y))
		

	
	D_trainXX = euclid_distance(X, X)	

	kernel_KXX = exp_kernel(D_trainXX, lam)
	extra = tf.matrix_inverse(kernel_KXX)
	R_train_gaus = gauss_reg(kernel_KXX, kernel)
	print kernel_KXX.get_shape()
	P_train_gaus = tf.transpose(tf.matmul(tf.transpose(R_train_gaus), Y))



	#print D_train.get_shape()
	#R_train = tf.cast(get_responsibilities(D_train, [size_of_set,80], k), tf.float64) 
	#print R_train.get_shape(), trainD.get_shape()
	#print R_train.get_shape(), Y

	#MSE = tf.reduce_sum(tf.pow(tf.subtract(P_train, tf.transpose(Y_T)),2)) / (2*size_of_set)
	#if (not returnPredictions):
	#	print sess.run(MSE, feed_dict = {X:trainData, X_T: comparisonData, Y: trainTarget, Y_T: comparisonTarget})
	#else:
	print sess.run(extra, feed_dict = {X:trainData, X_T: comparisonData, Y
    : trainTarget, Y_T: comparisonTarget})
	return sess.run(P_train_gaus, feed_dict = {X:trainData, X_T: comparisonData, Y: trainTarget, Y_T: comparisonTarget})

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

	with np.load ("data1D.npz") as data :

		print data
		trainData, trainTarget = data ["x"], data["y"]
		validData, validTarget = data ["x_valid"], data ["y_valid"]
		testData, testTarget = data ["x_test"], data ["y_test"]
		input()

	x = np.linspace(0.0, 11.0, num = 1000)[:,np.newaxis]
	for k in [1]:
		y = f(k, testData, testTarget, testTarget.size, True)
		a = open("a1_4_gauss.txt", "w")
		for b in y[0]:
			a.write(str(b) + "\n")
			print str(b)
		a.close()
		#plt.figure()
		#plt.plot(x,np.transpose(y))
		
	#plt.show()
		

plot()




