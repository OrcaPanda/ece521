import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from utils import *

def log_prob_dens_func(X, mu, sigma):
    B, D = X.get_shape().as_list()

    X = tf.expand_dims(X, 1)
    mu = tf.expand_dims(mu, 0)
    sigma = tf.expand_dims(sigma, 0)

    sX = tf.subtract(X, mu)
    return tf.multiply(tf.reduce_sum(tf.multiply(tf.square(sX), tf.reciprocal(tf.square(sigma))), 2), -0.5) - D * tf.log(2* pi) / 2 - tf.log(tf.reduce_prod(sigma))

def log_prob_clust_var(X, pi, mu, sigma):
    return log_prob_dens_func(X, mu, sigma) + tf.log(pi) - reduce_logsumexp(log_prob_dens_func(X,mu,sigma) + tf.log(pi), reduction_indices=0)

k = 3

epoch = 800
lr = 0.01

data = np.load("data2D.npy")
np.random.shuffle(data)

d = data.shape[1]
std = tf.Variable(tf.random_normal([k, d], stddev=0.01))
exp_std = tf.exp(std)

pi = tf.Variable(tf.random_normal([k,1], stddev=0.01))
soft_pi = logsoftmax(pi)

# Create a TF variable with normal-sampled initial values
mu = tf.Variable(tf.random_normal([k, d], stddev=0.01))

# x is K x B x D
x = tf.placeholder(tf.float32, [None, d])
L = tf.reduce_sum(reduce_logsumexp(log_prob_dens_func(x,mu,exp_std) + tf.log(soft_pi), reduction_indices=0))

train_step = tf.train.AdamOptimizer(lr, beta1=0.9, beta2=0.99, epsilon=1e-5).minimize(L)

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

B = len(data)
loss = []
for i in range(epoch):
    print("Epoch: ", i)
    sess.run(train_step, feed_dict={x: data})
    loss.append(sess.run(L, feed_dict={x: data}))

# Visualization
plot_x = list(range(len(loss)))
plt.plot(plot_x, loss, label='Loss per Update')
plt.xlabel('Updates')
plt.ylabel('Loss')
plt.title('Q2.2.2 Loss per Update')
plt.legend()
plt.show()