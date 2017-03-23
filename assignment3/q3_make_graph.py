import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pickle

import math

from utils import *

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
    print(variance.shape)
    ch  = 2.0 * tf.reduce_sum(tf.log(tf.diag_part(tf.cholesky(variance))))

    return (-D * tf.log(2*math.pi) - ch - eS) / 2 # B x K

k = 3
epoch = 400

np.random.seed(521)

data = np.load("tinymnist.npz")

lr = 0.01
D = 64

k = 4

W = tf.Variable(tf.random_normal([D, k], stddev=0.01))
W_pos = tf.exp(W)
phi_vec = tf.Variable(tf.random_normal([D], stddev=0.01))
phi_diag = tf.matrix_diag(tf.exp(phi_vec))

# Create a TF variable with normal-sampled initial values
mu = tf.Variable(tf.random_normal([k, D], stddev=0.01))

# x is B x D
X = tf.placeholder(tf.float32, [None, D])
L = -tf.reduce_sum(log_prob_dens_func(X, mu, tf.add(phi_diag, tf.matmul(W_pos, W_pos, transpose_b=True))))
likelihood = tf.reduce_sum(log_prob_dens_func(X, mu, tf.add(phi_diag, tf.matmul(W_pos, W_pos, transpose_b=True))))

train_step = tf.train.AdamOptimizer(lr, beta1=0.9, beta2=0.99, epsilon=1e-5).minimize(L)

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

valid_loss = []
test_loss = []
train_loss = []
for i in range(epoch):

    if i % 100 == 0:
        print(i)
    sess.run(train_step, feed_dict={X: data['x']})

    train_loss.append(sess.run(likelihood, feed_dict={X: data['x']}))
    valid_loss.append(sess.run(likelihood, feed_dict={X: data['x_valid']}))
    test_loss.append(sess.run(likelihood, feed_dict={X: data['x_test']}))


    # print("Epoch: ", i, loss[i])
    # if i % 100 == 0:5
    #     centers = sess.run(mu)
    #     plt.plot(data[:, 0], data[:, 1], 'b.')
    #     for i in range(len(centers)):
    #         plt.plot(centers[i][0], centers[i][1], 'k' + '.', markersize=20)
    #     plt.show()

centers = sess.run(mu)
phi = sess.run(phi_diag)
Weights = sess.run(W_pos)
pickle.dump((Weights, centers, phi, train_loss, valid_loss, test_loss), open("q3_400.pkl", "wb"))
for i in range(k):
    x = Weights[:, i]
    x = x.reshape((8, 8))
    plt.clf()
    plt.imshow(x)
    #plt.savefig("part4_" + str(i) + "_learning_rate_" + str(learning_rate) + ".png")
    plt.show()
