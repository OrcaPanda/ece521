###############################################################
# Part 1                                                      #
###############################################################

import tensorflow as tf
import numpy as np
import random
import sys
import pickle

import matplotlib
import matplotlib.pyplot as plt

###############################################################
# HELPER FUNCTIONS
###############################################################
def reformat(dataset, labels):
    dataset = dataset.reshape((-1, 28*28)).astype(np.float32)
    #labels = (np.arange(num_labels) == labels[:, None]).astype(np.float32)
    return dataset, labels

###############################################################
# IMPORTING TWO CLASS DATASET
###############################################################
with np.load("notMNIST.npz") as data :
    Data, Target = data ["images"], data["labels"]
    posClass = 2
    negClass = 9
    dataIndx = (Target == posClass) + (Target == negClass)
    Data = Data[dataIndx]/255.
    Target = Target[dataIndx].reshape(-1, 1)
    Target[Target == posClass] = 1
    Target[Target == negClass] = 0
    np.random.seed(521)
    randIndx = np.arange(len(Data))
    np.random.shuffle(randIndx)
    Data, Target = Data[randIndx], Target[randIndx]
    trainData, trainTarget = Data[:3500], Target[:3500]
    validData, validTarget = Data[3500:3600], Target[3500:3600]
    testData, testTarget = Data[3600:], Target[3600:]

    trainData, trainTarget = reformat(trainData, trainTarget)
    validData, validTarget = reformat(validData, validTarget)
    testData, testTarget = reformat(testData, testTarget)
    ###############################################################
    # SETTING UP HYPER PARAMETERS
    ###############################################################
    learning_rate = 0.0001
    epoch = 150
    batch_size = 500
    training_size = len(trainData)
    lam = 0.01

    convergence_percent = 0.005
    convergence_amount = 0.000000
    #Placeholders for data flow
    y_target = tf.placeholder(tf.float32, [None, 1])

    ###############################################################
    # ACTUAL CODE
    ###############################################################
    W = tf.Variable(tf.zeros([1, 784]))
    b = tf.Variable(0.0)

    y_pred = tf.placeholder(tf.float32, [None,1])
    l_d2 = tf.scalar_mul(0.5, tf.pow(tf.subtract(y_pred, y_target), 2))
    l_d1 = tf.nn.sigmoid_cross_entropy_with_logits(labels=y_target, logits=y_pred)

    #l_w = tf.scalar_mul(lam, tf.nn.l2_loss(W))
    #y_pred_sigmoid = tf.nn.sigmoid(y_pred)
    #l_cost = l_d #+ l_w

    #Define the gradient descent training step
    #train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(l_cost)
    #train_step = tf.train.AdamOptimizer(learning_rate).minimize(l_cost)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    loss_array_test = []
    accuracy_array_test = []

    loss_array_train = []
    accuracy_array_train = []

    loss_array_valid = []
    accuracy_array_valid = []


    y = np.linspace(0.0, 1.0, num=20)[:, np.newaxis]
    y_tar = np.zeros(20)[:, np.newaxis]

    loss1 = sess.run(l_d1, feed_dict={y_target: y_tar, y_pred: y})
    loss2 = sess.run(l_d2, feed_dict={y_target: y_tar, y_pred: y})
    print(y.shape)
    print(loss1)
    x = list(range(1, len(accuracy_array_test)+1))
    plt.plot(y, loss1, label = 'Logisitic' )
    plt.plot(y, loss2, label='Linear')
    plt.ylabel('Loss')
    plt.xlabel('Prediction')
    plt.title('Cross-entropy loss vs Squared-error loss')
    plt.legend()
    plt.show()

    print("hello")
