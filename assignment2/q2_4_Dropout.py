###############################################################
# Part 2.4
###############################################################

import tensorflow as tf
import numpy as np
import random
import sys

import matplotlib
import matplotlib.pyplot as plt
import pickle


def add_layer(inputTensor, hiddenUnits):
    inputSize = inputTensor.get_shape().as_list()[1]
    W = tf.get_variable(name="weights", shape=[inputSize, hiddenUnits], initializer=tf.contrib.layers.xavier_initializer(uniform=False))
    b = tf.get_variable(name="biases", shape=[1, hiddenUnits], initializer=tf.constant_initializer(0))
    return tf.add(tf.matmul(inputTensor, W), b)

###############################################################
# HELPER FUNCTIONS
###############################################################
def reformat(dataset, labels):
    dataset = dataset.reshape((-1, 28*28)).astype(np.float32)
    labels = (np.arange(10) == labels[:, None]).astype(np.float32)
    return dataset, labels

###############################################################
# IMPORTING TEN CLASS DATASET
###############################################################
with np.load("notMNIST.npz") as data:
    Data, Target = data ["images"], data["labels"]
    np.random.seed(521)
    randIndx = np.arange(len(Data))
    np.random.shuffle(randIndx)
    Data = Data[randIndx]/255.
    Target = Target[randIndx]
    trainData, trainTarget = Data[:15000], Target[:15000]
    validData, validTarget = Data[15000:16000], Target[15000:16000]
    testData, testTarget = Data[16000:], Target[16000:]

    trainData, trainTarget = reformat(trainData, trainTarget)
    validData, validTarget = reformat(validData, validTarget)
    testData, testTarget = reformat(testData, testTarget)

    ###############################################################
    # SETTING UP HYPER PARAMETERS
    ###############################################################
    learning_rate = 0.001
    epoch = 150
    batch_size = 500
    training_size = len(trainData)
    lam = 0.0003
    hidden_units = 1000
    savepoints = np.array([.25,.5,.75,1.])
    pickle_count = 0
    #saver = tf.train.Saver()

    x = tf.placeholder(tf.float32, [None, 784])
    y_target = tf.placeholder(tf.float32, [None, 10])

    ###############################################################
    # ACTUAL CODE
    ###############################################################

    with tf.variable_scope("layer_1"):
        z1_no_do = add_layer(x, hidden_units)
        z1_do = tf.nn.dropout(z1_no_do, 0.5)
        tf.get_variable_scope().reuse_variables()
        W1 = tf.get_variable("weights")

    with tf.variable_scope("layer_output"):
        y_do = add_layer(tf.nn.relu(z1_do), 10)
        tf.get_variable_scope().reuse_variables()
        W2 = tf.get_variable("weights")
        B2 = tf.get_variable("biases")
        y_out = tf.add(tf.matmul(z1_no_do, W2), B2)


    l_d = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_target, logits=y_do))

    l_w = tf.scalar_mul(lam, tf.add(tf.nn.l2_loss(W1), tf.nn.l2_loss(W2)))
    y_pred_softmax = tf.nn.softmax(logits=y_out)
    l_cost = l_d + l_w
    # Define the gradient descent training step
    train_step = tf.train.AdamOptimizer(learning_rate).minimize(l_cost)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    loss_array_test = []
    accuracy_array_test = []

    loss_array_train = []
    accuracy_array_train = []

    loss_array_valid = []
    accuracy_array_valid = []

    for i in range(epoch):
        print("epoch: ", i)
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

        ###############################################################
        # TEST SET
        ###############################################################
        compData, compTarget = testData, testTarget
        # Compute the cost
        sum_cost = sess.run(l_cost, feed_dict={x: compData, y_target: compTarget})
        loss_array_test.append(sum_cost)
        # Computing accuracy
        predictions = sess.run(y_pred_softmax, feed_dict={x: compData, y_target: compTarget})
        correct = 0.0
        count = 0
        for n in list(range(0, len(predictions))):
            if (np.argmax(predictions[n]) == np.argmax(compTarget[n])):
                correct += 1
            count += 1
        accuracy_array_test.append(correct / len(compTarget))

        ###############################################################
        # TRAINING SET
        ###############################################################
        compData, compTarget = trainData, trainTarget
        # Compute the cost
        sum_cost = sess.run(l_cost, feed_dict={x: compData, y_target: compTarget})
        loss_array_train.append(sum_cost)
        # Computing accuracy
        predictions = sess.run(y_pred_softmax, feed_dict={x: compData, y_target: compTarget})
        correct = 0.0
        count = 0
        for n in list(range(0, len(predictions))):
            if (np.argmax(predictions[n]) == np.argmax(compTarget[n])):
                correct += 1
            count += 1
        accuracy_array_train.append(correct / len(compTarget))

        ###############################################################
        # VALIDATION SET
        ###############################################################
        compData, compTarget = validData, validTarget
        # Compute the cost
        sum_cost = sess.run(l_cost, feed_dict={x: compData, y_target: compTarget})
        loss_array_valid.append(sum_cost)
        # Computing accuracy
        predictions = sess.run(y_pred_softmax, feed_dict={x: compData, y_target: compTarget})
        correct = 0.0
        count = 0
        for n in list(range(0, len(predictions))):
            if (np.argmax(predictions[n]) == np.argmax(compTarget[n])):
                correct += 1
            count += 1
        accuracy_array_valid.append(correct / len(compTarget))

        print(str(float(i + 1)))
        if float(i + 1) in savepoints * epoch:
            pickle_count += 1
            res = sess.run(W1)
            #pickle.dump(res, open("no_dropout" + str(pickle_count), "wb"))

        plt.interactive(False)

    parameters = learning_rate, 1, hidden_units, lam, 0.5
    total_results = accuracy_array_train, accuracy_array_valid, accuracy_array_test, loss_array_train, loss_array_valid, loss_array_test, parameters
    pickle.dump(total_results, open("q2_4_dropout_plot_data", "wb"));

    x = list(range(1, len(accuracy_array_test) + 1))
    plt.plot(x, accuracy_array_test, label='Test Data')
    plt.plot(x, accuracy_array_train, label='Training Data')
    plt.plot(x, accuracy_array_valid, label='Validation Data')
    plt.ylabel('Accuracy')
    plt.xlabel('Epochs')
    plt.title('notMNIST Accuracy lr-' + str(learning_rate))
    plt.legend(bbox_to_anchor=(.7, .8), loc=2, borderaxespad=0.)
    #plt.show()
    #plt.savefig("figure_q2_4_dropout_accuracy" + str(learning_rate) + ".png")
    #plt.clf()

    x = list(range(1, len(accuracy_array_test) + 1))
    plt.plot(x, loss_array_test, label='Test Data')
    plt.plot(x, loss_array_train, label='Training Data')
    plt.plot(x, loss_array_valid, label='Validation Data')
    plt.ylabel('Cross Entropy Loss')
    plt.xlabel('Epochs')
    plt.title('notMNIST Cross Entropy Loss lr-' + str(learning_rate))
    plt.legend(bbox_to_anchor=(.7, .8), loc=2, borderaxespad=0.)
    #plt.show()
    #plt.savefig("figure_q2_4_dropout_accuracy" + str(learning_rate) + ".png")

    print("hello")