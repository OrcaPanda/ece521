import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pickle
import math

def log_prob_dens_func(X, mu, variance):
    B, D = X.get_shape().as_list()

    X = tf.expand_dims(X, 1) # B x K x D
    mu = tf.expand_dims(mu, 0) # B x K x D
    n_variance = tf.expand_dims(variance, 0) # B x K x D x D
    n_variance = tf.expand_dims(n_variance, 0)  # B x K x D x D

    precision_matrix = tf.matrix_inverse(n_variance)

    sX = tf.subtract(X, mu) # B x K x D
    diff = tf.expand_dims(sX, 3)
    eS = tf.reduce_sum(tf.reduce_sum(tf.multiply( tf.multiply( diff, precision_matrix ),
                                                  tf.matrix_transpose(diff)), 2), 2)
    ch  = 2.0 * tf.reduce_sum(tf.log(tf.diag_part(tf.cholesky(variance))))

    return (-D * tf.log(2*math.pi) - ch - eS) / 2 # B x K

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

# Factor Analysis
epoch = 10000
lr = 0.001
mu = tf.Variable(tf.random_normal([1, D], stddev=0.01))
phi_vec = tf.Variable(tf.random_normal([D], stddev=0.01))
phi_diag = tf.matrix_diag(tf.exp(phi_vec))
W = tf.Variable(tf.random_normal([D, 1], stddev=0.01))
W_pos = tf.exp(W)
L = -tf.reduce_sum(log_prob_dens_func(X, mu, tf.add(phi_diag, tf.matmul(W_pos, W_pos, transpose_b=True))))
likelihood = -L
train_step = tf.train.AdamOptimizer(lr, beta1=0.9, beta2=0.99, epsilon=1e-5).minimize(L)
W_proj = tf.matmul(tf.matrix_inverse(tf.add(tf.matmul(W_pos, tf.matmul(tf.matrix_inverse(phi_diag), W_pos), transpose_a=True), 1.0)), tf.matmul(W_pos, tf.matrix_inverse(phi_diag), transpose_a=True))
fa_z = tf.matmul(X, W_proj, transpose_b=True)

# Running Tensorflow
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

# Getting Data for PCA
x = sess.run(X, feed_dict={S: data_S})
learnt_x = sess.run(pc_z, feed_dict={S: data_S})
slope = (np.max(learnt_x)-np.min(learnt_x)) / (np.max(x) - np.min(x))

# Getting Data for FA
for i in range(epoch):
	sess.run(train_step, feed_dict={S: data_S})
learnt_fa_x = sess.run(fa_z, feed_dict={S: data_S})
x1x2 = x[:, 0] + x[:, 1]
slope_fa = (np.max(learnt_fa_x)-np.min(learnt_fa_x)) / (np.max(x1x2) - np.min(x1x2)) * 2.0

# Plotting Results for PCA
plt.plot(x[:, 2], learnt_x, label='Slope: {:f}'.format(slope))
plt.xlabel('x3 Data')
plt.ylabel('PCA Learnt Direction Data')
plt.title('PCA with Single Principle Component')
plt.grid(True)
plt.legend()
plt.show()

# Plotting Results for FA
plt.plot(x[:, 0] + x[:, 1], learnt_fa_x, label='Slope: {:f}'.format(slope_fa))
plt.xlabel('x1 + x2 Data Normalized')
plt.ylabel('FA Learnt Direction Data')
plt.title('FA with Single Latent Dimension')
plt.grid(True)
plt.legend()
plt.show()