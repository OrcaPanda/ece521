###############################################################
# Part 2.5 Exhaustive Search
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

def neural_net(learning_rate, epoch, hidden_units, number_of_layers, weight_decay, keep_prob, batch_size):
    tf.reset_default_graph()
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
        training_size = len(trainData)

        x = tf.placeholder(tf.float32, [None, 784])
        y_target = tf.placeholder(tf.float32, [None, 10])

        kp = tf.placeholder(tf.float32, shape=())

        ###############################################################
        # ACTUAL CODE
        ###############################################################
        z = x
        all_w = []

        num = 0
        print(hidden_units)
        for units in hidden_units:
            with tf.variable_scope("layer_" + str(num)):
                z = tf.nn.relu(tf.nn.dropout(add_layer(z, units), kp))
                tf.get_variable_scope().reuse_variables()
                all_w.append(tf.get_variable("weights"))
            num += 1

        with tf.variable_scope("layer_output"):
            y = add_layer(z, 10)
            tf.get_variable_scope().reuse_variables()
            all_w.append(tf.get_variable("weights"))

        l_d = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_target, logits=y))

        accum_w = 0.
        for weight in all_w:
            accum_w = tf.add(accum_w, tf.nn.l2_loss(weight))

        l_w = tf.scalar_mul(weight_decay, accum_w)
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
            print("epoch: ", i, hidden_units, number_of_layers)
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
                sess.run(train_step, feed_dict={x: batch_xs, y_target: batch_ys, kp: keep_prob})

            ###############################################################
            # TEST SET
            ###############################################################
            compData, compTarget = testData, testTarget
            # Compute the cost
            sum_cost = sess.run(l_cost, feed_dict={x: compData, y_target: compTarget, kp: 1})
            loss_array_test.append(sum_cost)
            # Computing accuracy
            predictions = sess.run(y_pred_softmax, feed_dict={x: compData, y_target: compTarget, kp: 1})
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
            sum_cost = sess.run(l_cost, feed_dict={x: compData, y_target: compTarget, kp: 1})
            loss_array_train.append(sum_cost)
            # Computing accuracy
            predictions = sess.run(y_pred_softmax, feed_dict={x: compData, y_target: compTarget, kp: 1})
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
            sum_cost = sess.run(l_cost, feed_dict={x: compData, y_target: compTarget, kp: 1})
            loss_array_valid.append(sum_cost)
            # Computing accuracy
            predictions = sess.run(y_pred_softmax, feed_dict={x: compData, y_target: compTarget, kp: 1})
            correct = 0.0
            count = 0
            for n in list(range(0, len(predictions))):
                if (np.argmax(predictions[n]) == np.argmax(compTarget[n])):
                    correct += 1
                count += 1
            accuracy_array_valid.append(correct / len(compTarget))

        plt.interactive(False)

        return accuracy_array_test, accuracy_array_train, accuracy_array_valid, loss_array_test, loss_array_train, loss_array_valid

if __name__ == "__main__":
    ###############################################################
    # RUNNING FOR DIFFERENT PARAMETERS
    ###############################################################

    for index in (range(1)):
        epoch = 100
        batch_size = 500

<<<<<<< HEAD
        #learning_rate = np.exp((3.0)*np.random.random()-7.5)
        #number_of_layers = int(np.floor(5*np.random.random())+1)
        #number_of_hu = int(np.floor(401 * np.random.random()))+100
        #weight_decay = np.exp((3.0) * np.random.random() - 9.0)
        #keep_prob = 1.0
        #if(np.random.random() >= 0.5):
        #    keep_prob = 0.5
        learning_rate, number_of_layers, number_of_hu, weight_decay, keep_prob = 0.00333716, 2, 400, 0.00043443, 0.76
=======
        learning_rate = np.exp((3.0)*np.random.random()-7.5)
        number_of_layers = int(np.floor(5*np.random.random())+1)

        number_of_hu = []
        for k in range(number_of_layers):
            number_of_hu.append( int(np.floor(401 * np.random.random()))+100)
        weight_decay = np.exp((3.0) * np.random.random() - 9.0)
        keep_prob = 1.0
        if(np.random.random() >= 0.5):
            keep_prob = 0.5
        learning_rate, number_of_layers, number_of_hu, weight_decay, keep_prob = 0.00070512244750083291, 5, [236, 438, 326, 298, 496], 0.0017, 1.0
>>>>>>> 53ada55f6fcaa1013732295c5d40d2c9e9448e9a

        parameters = learning_rate, number_of_layers, number_of_hu, weight_decay, keep_prob
        print(parameters)

        accuracy_test = []
        accuracy_train = []
        accuracy_valid = []
        loss_test = []
        loss_train = []
        loss_valid = []

        hidden_units = number_of_hu

        results = neural_net(learning_rate, epoch, hidden_units, number_of_layers, weight_decay, keep_prob, batch_size)
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
        # plt.plot(x, 1 - np.array(accuracy_valid[0]), label='100 Hidden Units')
        # plt.ylabel('Error (%)')
        # plt.xlabel('Epochs')
        # plt.title('SGD notMNIST Error for Different Number of Hidden Units lr-' + str(learning_rate))
        # plt.legend()
        # plt.show()
	
        total_results = accuracy_train, accuracy_valid, accuracy_test, loss_train, loss_valid, loss_test, parameters
        pickle.dump( total_results, open("q2_5-"+str(learning_rate)+"_"+str(number_of_layers)+"_"+str(number_of_hu)+"_"+str(weight_decay)+"_"+str(keep_prob)+".p", "wb") )
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
