import tensorflow as tf
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pickle

k = 3

data = np.load("data2D.npy")

#######################################################################
#
# Question 1.2
#
#################################
#
# Hyperparameters
epoch = 800
lr = 0.01
#
#######################################################################
# Create a TF variable with normal-sampled initial values
mu = tf.Variable(tf.random_normal([k, data.shape[1]], stddev=0.01))
# x is K x B x D
x = tf.placeholder(tf.float32, [None, data.shape[1]])
# xu is now K x B
xu = tf.reduce_sum(tf.squared_difference(tf.expand_dims(mu,1), tf.expand_dims(x, 0)), axis = 2)
# xub is now B
xub = tf.reduce_min(xu, axis = 0)
# Finally, L is just a scalar
L = tf.reduce_sum(xub)

train_step = tf.train.AdamOptimizer(lr, beta1=0.9, beta2=0.99, epsilon=1e-5).minimize(L)

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

loss = []
for i in range(epoch):
    print("Epoch: ", i)
    sess.run(train_step, feed_dict={x: data})
    loss.append(sess.run(L, feed_dict={x: data}))

# Save the result
pickle.dump(loss, open("q1_loss.pkl", "wb"))

# Visualization
plot_x = list(range(len(loss)))
plt.plot(plot_x, loss, label='Loss per Update')
plt.xlabel('Updates')
plt.ylabel('Loss')
plt.title('Q1.1 Loss per Update')
plt.legend()
plt.show()
