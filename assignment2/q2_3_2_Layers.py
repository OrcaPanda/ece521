###############################################################
# Part 2.3.2 Different number of Layers
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

def neural_net(learning_rate, epoch, hidden_units):
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
        batch_size = 500
        training_size = len(trainData)
        lam = 0.0003

        x = tf.placeholder(tf.float32, [None, 784])
        y_target = tf.placeholder(tf.float32, [None, 10])

        ###############################################################
        # ACTUAL CODE
        ###############################################################

        with tf.variable_scope("layer_1"):
            z1 = add_layer(x, hidden_units)
            tf.get_variable_scope().reuse_variables()
            W1 = tf.get_variable("weights")
        with tf.variable_scope("layer_2"):
            z2 = add_layer(tf.nn.relu(z1), hidden_units)
            tf.get_variable_scope().reuse_variables()
            W2 = tf.get_variable("weights")
        with tf.variable_scope("layer_output"):
            y = add_layer(tf.nn.relu(z2), 10)
            tf.get_variable_scope().reuse_variables()
            W3 = tf.get_variable("weights")

        l_d = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_target, logits=y))

        l_w = tf.scalar_mul(lam, tf.add(tf.add(tf.nn.l2_loss(W1), tf.nn.l2_loss(W2)), tf.nn.l2_loss(W3)))
        y_pred_softmax = tf.nn.softmax(logits=y)
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

        plt.interactive(False)

        return accuracy_array_test, accuracy_array_train, accuracy_array_valid, loss_array_test, loss_array_train, loss_array_valid


    ###############################################################
    # RUNNING FOR DIFFERENT PARAMETERS
    ###############################################################
    learning_rate = 0.001
    epoch = 100;
    accuracy_test = []
    accuracy_train = []
    accuracy_valid = []
    loss_test = []
    loss_train = []
    loss_valid = []

    for hidden_units in (500):
        results = neural_net(learning_rate, epoch, hidden_units)
        accuracy_test.append(results[0])
        accuracy_train.append(results[1])
        accuracy_valid.append(results[2])
        loss_test.append(results[3])
        loss_train.append(results[4])
        loss_valid.append(results[5])

    ###############################################################
    # PLOTTING VALIDATION ERROR
    ###############################################################

    x = list(range(1, epoch + 1))
    plt.plot(x, 1 - np.array(accuracy_valid[0]), label='2 Layers, 500 Hidden Units Each')

    plt.ylabel('Error (%)')
    plt.xlabel('Epochs')
    plt.title('SGD notMNIST Error for Two Layers lr-' + str(learning_rate))
    plt.legend()
    plt.show()

    total_results = accuracy_train, accuracy_valid, accuracy_test, loss_train, loss_valid, loss_test
    pickle.dump(total_results, open("data2_3_2.p", "wb"))
    # x = list(range(1, len(accuracy_array_test) + 1))
    # plt.plot(x, accuracy_array_test, label='Test Data')
    # plt.plot(x, accuracy_array_train, label='Training Data')
    # plt.plot(x, accuracy_array_valid, label='Validation Data')
    # plt.ylabel('Accuracy (%)')
    # plt.xlabel('Epochs')
    # plt.title('SGD notMNIST Accuracy lr-' + str(learning_rate))
    # plt.legend(bbox_to_anchor=(.7, .8), loc=2, borderaxespad=0.)
    # plt.show()
    #
    # x = list(range(1, len(accuracy_array_test) + 1))
    # plt.plot(x, loss_array_test, label='Test Data')
    # plt.plot(x, loss_array_train, label='Training Data')
    # plt.plot(x, loss_array_valid, label='Validation Data')
    # plt.ylabel('Cross Entropy Loss')
    # plt.xlabel('Epochs')
    # plt.title('SGD notMNIST Cross Entropy Loss lr-' + str(learning_rate))
    # plt.legend(bbox_to_anchor=(.7, .8), loc=2, borderaxespad=0.)
    # plt.show()
    print("hello")