import tensorflow as tf
import numpy as np
import random
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def q2function(l_rate, batch_size, lam, classify = False):
	with np.load ("tinymnist.npz") as data :
		#Set up the data sets
		trainData, trainTarget = data ["x"], data["y"]
		validData, validTarget = data ["x_valid"], data ["y_valid"]
		testData, testTarget = data ["x_test"], data ["y_test"]

		#Hyperparameters and other parameters
		learning_rate = l_rate
		epoch = 300
		#batch_size = 700
		training_size = len(trainData)
		#print training_size
		#Define threshold for stopping
		convergence_percent = 0.005
		convergence_amount = 0.000000
		#Placeholders for data flow
		x = tf.placeholder(tf.float32, [None, 64])
		y_target = tf.placeholder(tf.float32, [None, 1])

		W = tf.Variable(tf.zeros([64, 1]))
		b = tf.Variable(0.0)
		#Weight-decay coefficient
		#lam = 1
		#Defines the loss function
		y_pre = tf.matmul(x,W)
		#print "y_pre", y_pre.get_shape()
		#b = tf.expand_dims(b, 0)
		#print "b", b.get_shape()
		y_predict = tf.add(y_pre,b)
		#print "y_predict", y_predict.get_shape()
		l_d = tf.scalar_mul(0.5,tf.reduce_mean(tf.pow(tf.subtract(y_predict, y_target),2)))
		#print l_d.get_shape()
		l_w = tf.scalar_mul( lam/2.0, tf.reduce_sum(tf.pow(W, 2)))
		l_cost = l_d + l_w
		#Define the gradient descent training step
		train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(l_cost)

		classify_accuracy = y_predict 

		sess = tf.Session()
		sess.run(tf.global_variables_initializer())

		prev_cost = 0.0001
		steps = 0

		loss_array = []

		for i in range(epoch):
			#print "epoch: ", i
			#Shuffle trainData and trainTarget in each epoch
			#combined = list(zip(trainData, trainTarget))
			#random.shuffle(combined)
			#shuf_trainData, shuf_trainTarget = trainData[:], trainTarget[:]
			#shuf_trainData[:], shuf_trainTarget[:] = zip(*combined)
			#Go through each of the batches within the epoch	
			
			rand_index = range(0,700)
			random.shuffle(rand_index)
			for k in range(0, training_size, batch_size):
				#Get the batches and run
				batch_xs, batch_ys = [],[]
				for val in rand_index[k:k+batch_size]:
					batch_xs.append(trainData[rand_index[val]])
					batch_ys.append(trainTarget[rand_index[val]])
				#trainData[rand_index[k : k+batch_size]], trainTarget[rand_index[k : k+batch_size]]
				#print len(batch_xs)
				#input()
				sess.run(train_step, feed_dict={x: batch_xs, y_target: batch_ys})
				#Compute the cost
				
			sum_cost = sess.run(l_cost, feed_dict={x: trainData, y_target: trainTarget})
			loss_array.append(sum_cost)
			#print sum_cost
			#Evaluate the stopping condition
			steps += 1
			if abs(prev_cost - sum_cost) < convergence_amount:
				prev_cost = sum_cost
				break
			#Otherwise store the result of the cost and continue
			prev_cost = sum_cost
		print "Steps taken: ", steps
		print "Final error: ", prev_cost
		
		f_write = open("2_2_1_l" + str(batch_size) + ".txt", "w")
		for val in loss_array:
			f_write.write(str(val) + "\n")
		
		if classify:
			predictions = sess.run(y_predict, feed_dict={x: testData, y_target: testTarget})
			correct = 0.0
			count = 0
			for n in range(len(predictions)):
				if predictions[n] > 0.5 and testTarget[n] > 0.5:
					correct += 1
				elif predictions[n] < 0.5 and testTarget[n] < 0.5:
					correct += 1
				count += 1
			#print "b", sess.run(b)
			print "lam: ", lam
			print "count", count
			print "correct: ", correct
			print "percent: ", correct/len(testTarget)


for i in [0.,0.0001,0.0004,0.0006,0.0008,0.001, 0.002,0.004,0.006,0.008,0.01,0.1,1.]:#0.001,0.005,0.01,0.05,0.1,0.2,0.3]:
	print "learning rate: ",i
	q2function(0.01,50,i,True)
