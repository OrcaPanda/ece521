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
    learning_rate = 0.001
    epoch = 250
    batch_size = 500
    training_size = len(trainData)
    lam = 0.01

    convergence_percent = 0.005
    convergence_amount = 0.000000
    #Placeholders for data flow
    x = tf.placeholder(tf.float32, [None, 784])
    y_target = tf.placeholder(tf.float32, [None, 1])

    ###############################################################
    # ACTUAL CODE
    ###############################################################
    W = tf.Variable(tf.zeros([1, 784]))
    b = tf.Variable(0.0)

    y_pred = tf.add(tf.matmul(x, tf.transpose(W)), b)
    l_d = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=y_target, logits=y_pred))
    l_w = tf.scalar_mul(lam, tf.nn.l2_loss(W))
    y_pred_sigmoid = tf.nn.sigmoid(y_pred)
    l_cost = l_d + l_w
    #Define the gradient descent training step
    #train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(l_cost)
    train_step = tf.train.AdamOptimizer(learning_rate).minimize(l_cost)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    loss_array_test = []
    accuracy_array_test = []

    loss_array_train = []
    accuracy_array_train = []

    for i in range(epoch):
        rand_index = list(range(0, len(trainData)))
        random.shuffle(rand_index)
        for k in range(0, training_size, batch_size):
            # Get the batches and run
            batch_xs, batch_ys = [], []
            for val in rand_index[k:k + batch_size]:
                batch_xs.append(trainData[rand_index[val]])
                batch_ys.append(trainTarget[rand_index[val]])
            # trainData[rand_index[k : k+batch_size]], trainTarget[rand_index[k : k+batch_size]]
            # print len(batch_xs)
            # input()

            batch_xs = np.array(batch_xs)
            batch_ys = np.array(batch_ys)
            sess.run(train_step, feed_dict={x: batch_xs, y_target: batch_ys})

        compData, compTarget = testData, testTarget
        # Compute the cost
        sum_cost = sess.run(l_cost, feed_dict={x: compData, y_target: compTarget})
        loss_array_test.append(sum_cost)
        # Computing accuracy
        predictions = sess.run(y_pred_sigmoid, feed_dict={x: compData, y_target: compTarget})
        correct = 0.0
        count = 0
        for n in list(range(0, len(predictions))):
            if predictions[n] > 0.5 and compTarget[n] > 0.5:
                correct += 1
            elif predictions[n] < 0.5 and compTarget[n] < 0.5:
                correct += 1
            count += 1

        accuracy_array_test.append(correct / len(compTarget))

        compData, compTarget = trainData, trainTarget
        # Compute the cost
        sum_cost = sess.run(l_cost, feed_dict={x: compData, y_target: compTarget})
        loss_array_train.append(sum_cost)
        # Computing accuracy
        predictions = sess.run(y_pred_sigmoid, feed_dict={x: compData, y_target: compTarget})
        correct = 0.0
        count = 0
        for n in list(range(0, len(predictions))):
            if predictions[n] > 0.5 and compTarget[n] > 0.5:
                correct += 1
            elif predictions[n] < 0.5 and compTarget[n] < 0.5:
                correct += 1
            count += 1

        accuracy_array_train.append(correct / len(compTarget))

    plt.interactive(False)

    pickle.dump(accuracy_array_train, open("q1_ADAM_training_accuracy","wb"))
    pickle.dump(accuracy_array_test, open("q1_ADAM_test_accuracy","wb"))
    pickle.dump(loss_array_train, open("q1_ADAM_training_loss","wb"))
    pickle.dump(loss_array_test, open("q1_ADAM_test_loss","wb"))

    x = list(range(1, len(accuracy_array_test)+1))
    plt.plot(x, accuracy_array_test, label = 'Test Data' )
    plt.plot(x, accuracy_array_train, label='Training Data')
    plt.ylabel('Accuracy (%)')
    plt.xlabel('Epochs')
    plt.title('SGD notMNIST Accuracy lr-0.001')
    plt.legend(bbox_to_anchor=(.7, .8), loc=2, borderaxespad=0.)
    plt.show()

    x = list(range(1, len(accuracy_array_test) + 1))
    plt.plot(x, loss_array_test, label='Test Data')
    plt.plot(x, loss_array_train, label='Training Data')
    plt.ylabel('Cross Entropy Loss')
    plt.xlabel('Epochs')
    plt.title('SGD notMNIST Cross Entropy Loss lr-0.001')
    plt.legend(bbox_to_anchor=(.7, .8), loc=2, borderaxespad=0.)
    plt.show()

    print("hello")