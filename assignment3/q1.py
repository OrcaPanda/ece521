import tensorflow as tf
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pickle

np.random.seed(521)

data = np.load("data2D.npy")
np.random.shuffle(data)
trainingData = data[0:6666]
validationData = data[6666:10000]
#######################################################################
#
# Hyperparameters
#
epoch = 1000
lr = 0.01
#
#######################################################################
for k in [1, 2, 3, 4, 5]:
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

    closest_k = tf.argmin(xu, axis = 0)

    train_step = tf.train.AdamOptimizer(lr, beta1=0.9, beta2=0.99, epsilon=1e-5).minimize(L)

    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)

    B = len(data)

    loss = []
    for i in range(epoch):
        print("Epoch: ", i)
        sess.run(train_step, feed_dict={x: trainingData})
        loss.append(sess.run(L, feed_dict={x: validationData}))

    centers = sess.run(mu)
    k_distribution = sess.run(closest_k, feed_dict={x: validationData})
    clustered_results = []
    for i in range(k):
        clustered_results.append([])

    index = 0
    for which_k in k_distribution:
        clustered_results[which_k].append(validationData[index])
        index += 1

    # Save the result
    # pickle.dump((loss, centers, k_distribution), open("q2_2_4_loss_k_" + str(k) + ".pkl", "wb"))

    # Visualization
    plot_x = list(range(len(loss)))
    plt.plot(plot_x, loss, label='Loss per Update')
    plt.xlabel('Updates')
    plt.ylabel('Loss')
    plt.title('Q2.2.3 Loss per Update')
    plt.legend()
    #plt.show()
    
    colors = ('g', 'r', 'b', 'y', 'm')
    for i in range(k):
        a = np.array(clustered_results[i])
        percentage = len(a) / B * 100.
        lbl = 'Cluster {}: {:0.2f}% of data points'.format(str(i + 1), percentage)
        plt.plot( a[:,0], a[:,1], colors[i] + '.', label= lbl )
        plt.plot(centers[i][0], centers[i][1], 'k' + '.', markersize=20)
    plt.grid(True)
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.title('Q1.4 Clustered results - Validation dataset')
    plt.legend()
    #plt.show()
    print("Loss for k = " + str(k) + " : " + str(loss[-1]))