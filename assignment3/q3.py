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
    variance = tf.expand_dims(variance, 0) # B x K x D x D

    precision_matrix = tf.matrix_inverse(variance)

    sX = tf.subtract(X, mu) # B x K x D
    diff = tf.expand_dims(sX, 3)
    eS = tf.reduce_sum(tf.reduce_sum(tf.multiply( tf.multiply( diff, precision_matrix ),
                                                  tf.matrix_transpose(diff)), 2), 2)
    ch  = 2.0 * tf.reduce_sum(tf.log(tf.diag_part(tf.cholesky(variance))))

    return (-D * tf.log(2*math.pi) - ch - eS) / 2 # B x K

def log_prob_clust_var(X, pi, mu, variance):
    # pi is 1 x K
    rlse = reduce_logsumexp(log_prob_dens_func(X,mu,variance) + tf.log(pi), reduction_indices=1) # B
    rlse = tf.expand_dims(rlse, 1) # B x K
    return log_prob_dens_func(X, mu, variance) + tf.log(pi) - rlse # B x K

k = 3
epoch = 10000

np.random.seed(521)

data = np.load("data100D.npy")
np.random.shuffle(data)
trainingData = data[0:6666]
validationData = data[6666:10000]
lr = 0.01
D = data.shape[1]

for k in range(1, 16):

    phi = tf.Variable(tf.random_normal([k, D], stddev=0.01))
    exp_var = tf.exp(phi) # K x D

    psi = tf.Variable(tf.random_normal([1, k], stddev=0.01))
    soft_pi = logsoftmax(psi) # 1 x K

    # Create a TF variable with normal-sampled initial values
    mu = tf.Variable(tf.random_normal([k, D], stddev=0.01))

    # x is B x D
    X = tf.placeholder(tf.float32, [None, D])
    L = -tf.reduce_sum(reduce_logsumexp(log_prob_dens_func(X,mu,exp_var) + soft_pi, reduction_indices=1))

    train_step = tf.train.AdamOptimizer(lr, beta1=0.9, beta2=0.99, epsilon=1e-5).minimize(L)

    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)

    print(log_prob_dens_func(X,mu,exp_var).shape)

    B = len(data)
    loss = []
    for i in range(epoch):
        sess.run(train_step, feed_dict={X: trainingData})
        loss.append(sess.run(L, feed_dict={X: validationData}))

        # print("Epoch: ", i, loss[i])
        # if i % 100 == 0:
        #     centers = sess.run(mu)
        #     plt.plot(data[:, 0], data[:, 1], 'b.')
        #     for i in range(len(centers)):
        #         plt.plot(centers[i][0], centers[i][1], 'k' + '.', markersize=20)
        #     plt.show()

    centers = sess.run(mu)
    std = sess.run(exp_var)
    pickle.dump((loss, centers, std), open("q2_2_100D_" + str(k) +".pkl", "wb"))

    # print(sess.run(mu))
    # k = sess.run(tf.exp(log_prob_dens_func(X,mu[0],exp_var)), feed_dict={X: sess.run(mu)})
    # print(k.shape)
    # print(k)
    # sum0, sum1, sum2 = 0,0,0
    # for x in k:
    #
    #     sum0 += x[0]
    #     sum1 += x[1]
    #     sum2 += x[2]
    # print(sum0, sum1, sum2)

    k_distribution = sess.run(tf.argmax(log_prob_dens_func(X,mu,exp_var) + soft_pi, axis=1), feed_dict={X: validationData})

    clustered_results = []
    for i in range(k):
        clustered_results.append([])

    index = 0
    for which_k in k_distribution:
        clustered_results[which_k].append(validationData[index])
        index += 1


    colors = ('g', 'r', 'b', 'y', 'm')
    for i in range(k):
        a = np.array(clustered_results[i])
        percentage = len(a) / len(validationData) * 100.
        lbl = 'Cluster {}: {:0.2f}% of data points'.format(str(i + 1), percentage)
        if len(a)!=0:
            plt.plot( a[:,0], a[:,1], colors[i] + '.', label= lbl )
        plt.plot(centers[i][0], centers[i][1], 'k' + '.', markersize=20)
    plt.grid(True)
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.title('Q2.2.2 ' + str(k) + ' Clustered results - Validation dataset ' )
    plt.legend()
    plt.savefig('2_2_2_K_' + str(k) + '.png')
    plt.clf()
    print("Loss for k = " + str(k) + " : " + str(loss[-1]))
