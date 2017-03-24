import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pickle
import math

# Dimensions of dataset
B = 200
D = 3

# Generate the latent states from a Gaussian
data_S = np.random.normal(0.0, 1.0, [B, D])

# Generate the toy dataset
S = tf.placeholder(tf.float32, [B, D])
X = tf.matmul(S, tf.transpose(tf.constant([[1, 0, 0], [1, 0.001, 0], [0, 0, 10]])))

# PCA
X_mean = tf.reduce_mean(X, 0)
X_diff = tf.expand_dims(X - X_mean, 2)
X_cov = tf.reduce_mean(tf.matmul(X_diff, tf.matrix_transpose(X_diff)), 0)
# Returned eigenvectors are already normalized
e_X, v_X = tf.self_adjoint_eig(X_cov)
# Project each x into the subspace. The largest eigenvalue is last
pc = tf.slice(tf.multiply(v_X, -1.0), [0, 2], [3, 1])
pc_z = tf.matmul(X, pc)

# Running Tensorflow
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

# Getting Data
x = sess.run(X, feed_dict={S: data_S})
learnt_x = sess.run(pc_z, feed_dict={S: data_S})
slope = (np.max(learnt_x)-np.min(learnt_x)) / (np.max(x) - np.min(x))

# Plotting Results
plt.plot(x[:, 2], learnt_x, label='Slope: {:f}'.format(slope))
plt.ylabel('x3 Data')
plt.xlabel('PCA Learnt Direction Data')
plt.title('PCA with Single Principle Component')
plt.grid(True)
plt.legend()
plt.show()