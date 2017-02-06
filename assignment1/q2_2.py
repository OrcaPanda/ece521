import tensorflow as tf
import numpy as np
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

with np.load ("tinymnist.npz") as data :
	trainData, trainTarget = data ["x"], data["y"]
	validData, validTarget = data ["x_valid"], data ["y_valid"]
	testData, testTarget = data ["x_test"], data ["y_test"]

x = tf.placeholder(tf.float32, [None, 64])

W = tf.Variable(tf.zeros([64, 1]))
b = tf.Variable(tf.zeros([2]))
